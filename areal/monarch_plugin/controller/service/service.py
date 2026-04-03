"""Service controller: manages replicated actors with health monitoring and load balancing.

Ported from TorchForge's Service with identical semantics:
- N replicas, each an independent ActorMesh
- Health loop detects failed replicas and triggers recovery
- Request routing via configurable Router strategy
- Session-based affinity for stateful workloads (agentic RL)
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from areal.monarch_plugin.controller.service.interface import (
    Session,
    _session_context,
)
from areal.monarch_plugin.controller.service.metrics import ServiceMetrics
from areal.monarch_plugin.controller.service.replica import Replica, ServiceRequest
from areal.monarch_plugin.controller.service.router import (
    LeastLoadedRouter,
    RoundRobinRouter,
    SessionRouter,
)
from areal.monarch_plugin.types import ServiceConfig

logger = logging.getLogger(__name__)


class Service:
    """Manages multiple actor replicas with automatic fault tolerance.

    Args:
        cfg: Service configuration.
        actor_def: Actor class to instantiate per replica.
        actor_args: Positional args for the actor constructor.
        actor_kwargs: Keyword args for the actor constructor.
    """

    def __init__(
        self,
        cfg: ServiceConfig,
        actor_def,
        actor_args: tuple,
        actor_kwargs: dict,
    ):
        self._cfg = cfg
        self._replicas: list[Replica] = []
        self._actor_def = actor_def
        self._actor_args = actor_args
        self._actor_kwargs = actor_kwargs

        self._active_sessions: list[Session] = []
        self._session_replica_map: dict[str, int] = {}

        self._metrics = ServiceMetrics()
        self._health_task: asyncio.Task | None = None
        self._shutdown_requested = False
        self._replicas_to_recover: list[Replica] = []

    async def __initialize__(self):
        """Create all replicas and start the health loop."""
        logger.debug(f"Starting service with {self._cfg.num_replicas} replicas.")

        self._default_router = RoundRobinRouter()
        self._session_router = SessionRouter(fallback_router=LeastLoadedRouter())

        replicas = []
        for i in range(self._cfg.num_replicas):
            replica = Replica(
                idx=i,
                proc_config=self._cfg.to_process_config(),
                max_concurrent_requests=self._cfg.replica_max_concurrent_requests,
                return_first_rank_result=self._cfg.return_first_rank_result,
                actor_def=self._actor_def,
                actor_args=self._actor_args,
                actor_kwargs=self._actor_kwargs,
            )
            replicas.append(replica)

        await asyncio.gather(*[r.initialize() for r in replicas])
        self._replicas = replicas

        self._health_task = asyncio.create_task(
            self._health_loop(poll_rate_s=self._cfg.health_poll_rate)
        )

    async def _call(self, sess_id: str | None, function: str, *args, **kwargs):
        """Route a call to a replica with load balancing and retry."""
        if sess_id is None:
            ctx = _session_context.get(None)
            if ctx:
                sess_id = ctx["session_id"]

        replica = await self._get_replica(sess_id)

        request = ServiceRequest(
            session_id=sess_id,
            function=function,
            args=args,
            kwargs=kwargs,
            future=asyncio.Future(),
        )
        await replica.enqueue_request(request)

        try:
            return await request.future
        except Exception:
            if not replica.healthy:
                logger.debug(
                    f"Replica {replica.idx} failed, retrying on healthy replica."
                )
                return await self._retry_on_healthy(sess_id, function, *args, **kwargs)
            raise

    async def call_all(self, function: str, *args, **kwargs) -> list:
        """Broadcast a call to all healthy replicas."""
        healthy = [r for r in self._replicas if r.healthy]
        if not healthy:
            raise RuntimeError("No healthy replicas available for broadcast")

        requests = []
        for replica in healthy:
            request = ServiceRequest(
                session_id=None,
                function=function,
                args=args,
                kwargs=kwargs,
                future=asyncio.Future(),
            )
            requests.append((replica, request))

        for replica, request in requests:
            await replica.enqueue_request(request)

        results = []
        for replica, request in requests:
            try:
                result = await request.future
                results.append(result)
            except Exception as e:
                logger.warning(f"Broadcast to replica {replica.idx} failed: {e}")
                results.append(None)
        return results

    async def _retry_on_healthy(
        self, sess_id: str | None, function: str, *args, **kwargs
    ):
        if sess_id is not None and sess_id in self._session_replica_map:
            del self._session_replica_map[sess_id]
        return await self._call(sess_id, function, *args, **kwargs)

    async def start_session(self) -> str:
        sess_id = str(uuid.uuid4())
        self._active_sessions.append(Session(session_id=sess_id))
        self._update_metrics()
        return sess_id

    async def terminate_session(self, sess_id: str):
        self._active_sessions = [
            s for s in self._active_sessions if s.session_id != sess_id
        ]
        self._session_replica_map.pop(sess_id, None)
        self._update_metrics()

    def _update_metrics(self):
        self._metrics.total_sessions = len(self._active_sessions)
        self._metrics.total_replicas = len(self._replicas)
        self._metrics.healthy_replicas = sum(1 for r in self._replicas if r.healthy)
        self._metrics.replica_metrics = {r.idx: r.metrics for r in self._replicas}

    def get_metrics(self) -> ServiceMetrics:
        self._update_metrics()
        return self._metrics

    def get_metrics_summary(self) -> dict:
        self._update_metrics()
        summary = {
            "service": {
                "total_sessions": self._metrics.total_sessions,
                "healthy_replicas": self._metrics.healthy_replicas,
                "total_replicas": self._metrics.total_replicas,
            },
            "replicas": {},
        }
        for replica in self._replicas:
            summary["replicas"][replica.idx] = {
                "total_requests": replica.metrics.total_requests,
                "active_requests": replica.active_requests,
                "queue_depth": replica.qsize(),
                "capacity_utilization": replica.capacity_utilization,
            }
        return summary

    async def _get_replica(self, sess_id: str | None) -> Replica:
        healthy = [r for r in self._replicas if r.healthy]
        if sess_id is None:
            return self._default_router.get_replica(healthy)
        return self._session_router.get_replica(
            healthy, sess_id, self._session_replica_map
        )

    async def _health_loop(self, poll_rate_s: float):
        while not self._shutdown_requested:
            await self._recover_replicas()
            failed = [r for r in self._replicas if r.failed]
            if failed:
                logger.debug(f"[HEALTH] {len(failed)} failed replica(s) detected")
                self._replicas_to_recover.extend(failed)
            await asyncio.sleep(poll_rate_s)

    async def _recover_replicas(self):
        if not self._replicas_to_recover:
            return

        async def _recover(replica):
            try:
                await replica.recover()
            except Exception as e:
                logger.error(f"Failed to recover replica {replica.idx}: {e}")
                replica.mark_failed()

        tasks = [asyncio.create_task(_recover(r)) for r in self._replicas_to_recover]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._replicas_to_recover.clear()

    async def stop(self):
        """Stop the service: cancel health loop, stop all replicas."""
        self._shutdown_requested = True
        if self._health_task is not None:
            try:
                await asyncio.wait_for(self._health_task, timeout=5.0)
            except TimeoutError:
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass

        await asyncio.gather(
            *[r.stop() for r in self._replicas], return_exceptions=True
        )

    async def _get_internal_state(self) -> dict:
        """Testing/debugging only."""
        self._update_metrics()
        return {
            "session_replica_map": dict(self._session_replica_map),
            "active_sessions": [s.session_id for s in self._active_sessions],
            "replicas": [
                {
                    "idx": r.idx,
                    "state": r.state.value,
                    "healthy": r.healthy,
                    "active_requests": r.active_requests,
                }
                for r in self._replicas
            ],
            "total_replicas": len(self._replicas),
            "shutdown_requested": self._shutdown_requested,
        }
