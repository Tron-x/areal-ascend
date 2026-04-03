"""
TorchForge-inspired Service Layer for AReaL Monarch Plugin.

A Service object manages multiple homogeneous replicas (e.g. multiple GeneratorActors)
and routes incoming requests across them using dynamic load balancing.
It also provides isolation and self-healing.
"""

import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("MonarchService")

@dataclass
class ServiceRequest:
    session_id: str | None
    function: str
    args: tuple
    kwargs: dict
    future: asyncio.Future


class Replica:
    def __init__(self, idx: int, actor_ref: Any):
        self.idx = idx
        self.actor_ref = actor_ref
        self.is_healthy = True
        self.in_flight_requests = 0
        self._recovery_task = None
        
    async def _check_health(self) -> bool:
        """Attempt to recover by calling a simple health endpoint."""
        try:
            # All AReaLMonarchActors now have a universal 'health' endpoint.
            # Using asyncio.wrap_future to safely await the underlying Monarch Future
            future = self.actor_ref.health.call()
            import concurrent.futures
            if isinstance(future, concurrent.futures.Future):
                await asyncio.wrap_future(future)
            else:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: future.get(timeout=10)
                )
            return True
        except Exception:
            return False

    async def _recovery_loop(self) -> None:
        """Background loop to attempt recovery of an unhealthy replica."""
        while not self.is_healthy:
            await asyncio.sleep(5)  # Retry every 5s
            if await self._check_health():
                self.is_healthy = True
                logger.info(f"Replica {self.idx} has recovered and is now healthy.")
                self._recovery_task = None
                return

    def mark_unhealthy(self) -> None:
        self.is_healthy = False
        if self._recovery_task is None:
            self._recovery_task = asyncio.create_task(self._recovery_loop())

        
    async def enqueue_request(self, request: ServiceRequest) -> None:
        if not self.is_healthy:
            request.future.set_exception(RuntimeError(f"Replica {self.idx} is unhealthy"))
            return
            
        self.in_flight_requests += 1
        try:
            # actor_ref is a Monarch RPC reference
            func = getattr(self.actor_ref, request.function)
            result = await func.call(*request.args, **request.kwargs)
            if not request.future.done():
                request.future.set_result(result)
        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)
            self.mark_unhealthy()
            logger.error(f"Replica {self.idx} failed and is marked unhealthy: {e}")
        finally:
            self.in_flight_requests -= 1


class Router(abc.ABC):
    @abc.abstractmethod
    def get_replica(self, session_id: str | None, replicas: list[Replica]) -> Replica:
        pass


class LeastLoadedRouter(Router):
    def __init__(self, max_concurrency: int = 16):
        self.max_concurrency = max_concurrency

    def get_replica(self, session_id: str | None, replicas: list[Replica]) -> Replica:
        healthy_replicas = [r for r in replicas if r.is_healthy]
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available!")
            
        # Sort by load
        best = min(healthy_replicas, key=lambda r: r.in_flight_requests)
        
        # Simple backpressure
        if best.in_flight_requests >= self.max_concurrency:
            logger.warning(f"All replicas are at max capacity ({self.max_concurrency}). Potential bottleneck.")
            
        return best


class Service:
    """
    Manages a pool of homogeneous Actor references created by `.as_service()`
    and routes RPC requests to them.
    """
    def __init__(self, actor_refs: list[Any], actor_cls: type, router_type: str = "least_loaded"):
        self.actor_cls = actor_cls
        self.actor_refs = actor_refs
        self._replicas = [Replica(i, ref) for i, ref in enumerate(actor_refs)]
        
        if router_type == "least_loaded":
            self._router = LeastLoadedRouter()
        else:
            self._router = LeastLoadedRouter()
            
        logger.info(f"Service initialized with {len(self._replicas)} replicas.")

    async def call(self, function: str, *args, session_id: str | None = None, **kwargs) -> Any:
        request = ServiceRequest(
            session_id=session_id,
            function=function,
            args=args,
            kwargs=kwargs,
            future=asyncio.Future()
        )
        
        replica = self._router.get_replica(session_id, self._replicas)
        # We don't await this directly, fire and forget enqueue which satisfies the future
        asyncio.create_task(replica.enqueue_request(request))
        return await request.future

    async def call_all(self, function: str, *args, **kwargs) -> list[Any]:
        """Broadcast call to all healthy replicas. Raises if any replica fails."""
        healthy_replicas = [r for r in self._replicas if r.is_healthy]
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available to broadcast.")
            
        futures = []
        for replica in healthy_replicas:
            request = ServiceRequest(
                session_id=None,
                function=function,
                args=args,
                kwargs=kwargs,
                future=asyncio.Future()
            )
            asyncio.create_task(replica.enqueue_request(request))
            futures.append(request.future)
            
        # Use return_exceptions=True to catch individual errors and aggregate them
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.error(f"Broadcasting {function} failed on {len(errors)} replicas. First error: {errors[0]}")
            raise RuntimeError(f"Service broadcast failed: {errors[0]}") from errors[0]
            
        return results
