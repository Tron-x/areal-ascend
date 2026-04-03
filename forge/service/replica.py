"""Replica lifecycle management for distributed actor services."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from monarch.actor import ActorError

if TYPE_CHECKING:
    from forge.actors.base import ForgeActor
    from forge.types import ProcessConfig

logger = logging.getLogger(__name__)


class ReplicaState(Enum):
    HEALTHY = "HEALTHY"
    RECOVERING = "RECOVERING"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"
    UNINITIALIZED = "UNINITIALIZED"


@dataclass
class ReplicaMetrics:
    """Per-replica request tracking."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))
    request_latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_request_start(self, timestamp: float):
        self.request_times.append(timestamp)
        self.total_requests += 1

    def add_request_completion(self, start_time: float, success: bool):
        latency = time.time() - start_time
        self.request_latencies.append(latency)
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def get_request_rate(self, window_seconds: float = 60.0) -> float:
        now = time.time()
        cutoff = now - window_seconds
        recent = [t for t in self.request_times if t >= cutoff]
        return len(recent) / window_seconds

    def get_avg_latency(self, window_requests: int = 50) -> float:
        if not self.request_latencies:
            return 0.0
        recent = list(self.request_latencies)[-window_requests:]
        return sum(recent) / len(recent)


@dataclass
class ServiceRequest:
    """A queued request to a service replica."""

    session_id: str | None
    function: str
    args: tuple
    kwargs: dict
    future: asyncio.Future


@dataclass
class Replica:
    """A single replica within a distributed service.

    Handles process lifecycle, async request queuing, and fault recovery.
    """

    idx: int
    proc_config: ProcessConfig
    actor_def: type[ForgeActor]
    actor_args: tuple
    actor_kwargs: dict

    actor: ForgeActor | None = None
    request_queue: asyncio.Queue[ServiceRequest] = field(default_factory=asyncio.Queue)
    active_requests: int = 0
    max_concurrent_requests: int = 10
    _capacity_semaphore: asyncio.Semaphore = field(init=False)
    _running: bool = False
    _run_poll_rate_s: float = 1.0
    state: ReplicaState = ReplicaState.UNINITIALIZED
    return_first_rank_result: bool = False
    _recovery_task: asyncio.Task | None = None
    _run_task: asyncio.Task | None = None
    metrics: ReplicaMetrics = field(default_factory=ReplicaMetrics)
    _base_mesh_name: str | None = field(default=None, init=False)
    _mesh_name_initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        self._capacity_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self._base_mesh_name = self.proc_config.mesh_name

    async def initialize(self):
        """Initialize the replica: launch actor, start processing loop."""
        assert self.actor is None, "Actor should not be set yet"
        try:
            if self._base_mesh_name and not self._mesh_name_initialized:
                mesh_name_with_replica = f"{self._base_mesh_name}_{self.idx}"
                self.proc_config.mesh_name = mesh_name_with_replica
                if hasattr(self.actor_def, "mesh_name"):
                    self.actor_def.mesh_name = mesh_name_with_replica
                self._mesh_name_initialized = True

            self.actor = await self.actor_def.launch(
                *self.actor_args, **self.actor_kwargs
            )
            self.state = ReplicaState.HEALTHY
            self.start_processing()
            logger.debug(f"Replica {self.idx} initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize replica {self.idx}: {e}")
            self.state = ReplicaState.UNHEALTHY
            raise

    async def recover(self):
        """Recover by shutting down and re-initializing."""
        if self._recovery_task and not self._recovery_task.done():
            await self._recovery_task
            return

        async def _do_recovery():
            try:
                await self.actor_def.shutdown(self.actor)
                self.actor = None
            except Exception as e:
                logger.warning(f"Error shutting down actor for replica {self.idx}: {e}")
                self.state = ReplicaState.UNHEALTHY
            try:
                await self.initialize()
            except Exception as e:
                logger.error(f"Recovery failed for replica {self.idx}: {e}")
                self.state = ReplicaState.UNHEALTHY
                raise

        self.state = ReplicaState.RECOVERING
        self._recovery_task = asyncio.create_task(_do_recovery())
        await self._recovery_task

    def start_processing(self):
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.run())

    async def enqueue_request(self, request: ServiceRequest):
        if self.stopped:
            raise RuntimeError(
                f"Replica {self.idx} is stopped and will not accept requests."
            )
        await self.request_queue.put(request)

    async def _process_single_request(self, request: ServiceRequest) -> bool:
        start_time = time.time()
        self.active_requests += 1
        self.metrics.add_request_start(start_time)

        try:
            actor = self.actor
            endpoint_func = getattr(actor, request.function)
            success = True
            try:
                result = await endpoint_func.call(*request.args, **request.kwargs)
                if self.return_first_rank_result:
                    _, first_result = next(result.items())
                    result = first_result
                request.future.set_result(result)
            except ActorError as e:
                logger.warning(f"Actor error on replica {self.idx}: {e}")
                request.future.set_result(e.exception)
                self.mark_failed()
                success = False
            except Exception as e:
                logger.debug(f"Unexpected error on replica {self.idx}: {e}")
                self.mark_failed()
                request.future.set_exception(e)
                success = False

            self.metrics.add_request_completion(start_time, success)
            self.request_queue.task_done()
            return success
        finally:
            self.active_requests -= 1
            self._capacity_semaphore.release()

    async def run(self):
        """Main processing loop: dequeue and dispatch requests."""
        self._running = True
        try:
            while self.healthy:
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self._run_poll_rate_s
                    )
                    await self._capacity_semaphore.acquire()
                    asyncio.create_task(self._process_single_request(request))
                except TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in replica {self.idx} processing loop: {e}")
                    self.state = ReplicaState.UNHEALTHY
                    break
        finally:
            self._running = False

    @property
    def healthy(self) -> bool:
        return self.state == ReplicaState.HEALTHY

    @property
    def stopped(self) -> bool:
        return self.state == ReplicaState.STOPPED

    @property
    def failed(self) -> bool:
        return self.state in (ReplicaState.RECOVERING, ReplicaState.UNHEALTHY)

    def mark_failed(self):
        self.state = ReplicaState.RECOVERING

    async def stop(self):
        """Gracefully stop: cancel run loop, fail pending requests, shutdown actor."""
        self.state = ReplicaState.STOPPED
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await asyncio.wait_for(
                    self._run_task, timeout=2 * self._run_poll_rate_s
                )
            except (TimeoutError, asyncio.CancelledError):
                pass

        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError(f"Replica {self.idx} is stopping")
                    )
                self.request_queue.task_done()
            except asyncio.QueueEmpty:
                break

        if self.actor:
            try:
                await self.actor_def.shutdown(self.actor)
            except Exception as e:
                logger.warning(f"Error stopping proc_mesh for replica {self.idx}: {e}")

    @property
    def current_load(self) -> int:
        return self.active_requests + self.request_queue.qsize()

    def qsize(self) -> int:
        return self.request_queue.qsize()

    @property
    def capacity_utilization(self) -> float:
        if self.max_concurrent_requests <= 0:
            return 0.0
        return self.active_requests / self.max_concurrent_requests

    def __repr__(self) -> str:
        return (
            f"Replica(idx={self.idx}, state={self.state.value}, "
            f"active={self.active_requests}/{self.max_concurrent_requests}, "
            f"queue={self.request_queue.qsize()})"
        )
