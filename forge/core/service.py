"""Framework-agnostic Service layer for Forge.

Manages multiple replicas of a ForgeActor, routes requests via pluggable
routers, and monitors replica health. Pure asyncio -- no Monarch or Ray.

Extracted from ``areal/monarch_plugin/service/service.py`` and generalised
to use an ``ActorRef`` protocol instead of Monarch-specific references.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("forge.service")


@runtime_checkable
class ActorRef(Protocol):
    """Minimal protocol for a remote actor reference.

    Any RPC framework (Monarch, Ray, gRPC) can satisfy this by providing
    attribute access to remote methods that are callable.
    """

    def __getattr__(self, name: str) -> Any: ...


@dataclass
class ServiceRequest:
    """Internal: tracks a pending request to a replica."""

    session_id: str | None
    function: str
    args: tuple
    kwargs: dict
    future: asyncio.Future


class Replica:
    """Manages a single actor instance within a Service."""

    def __init__(self, idx: int, actor_ref: ActorRef) -> None:
        self.idx = idx
        self.actor_ref = actor_ref
        self.is_healthy = True
        self.in_flight: int = 0
        self._recovery_task: asyncio.Task | None = None

    def mark_unhealthy(self) -> None:
        self.is_healthy = False
        if self._recovery_task is None:
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def _recovery_loop(self) -> None:
        while not self.is_healthy:
            await asyncio.sleep(5)
            try:
                ref = getattr(self.actor_ref, "health", None)
                if ref is not None:
                    call = getattr(ref, "call", ref)
                    if asyncio.iscoroutinefunction(call):
                        await call()
                    else:
                        await asyncio.get_running_loop().run_in_executor(None, call)
                self.is_healthy = True
                logger.info("Replica %d recovered", self.idx)
                self._recovery_task = None
                return
            except Exception:
                pass

    async def dispatch(self, request: ServiceRequest) -> None:
        """Dispatch a request to this replica."""
        if not self.is_healthy:
            request.future.set_exception(
                RuntimeError(f"Replica {self.idx} is unhealthy")
            )
            return

        self.in_flight += 1
        try:
            func = getattr(self.actor_ref, request.function)
            call_method = getattr(func, "call", None) or getattr(func, "call_one", None)
            if call_method is not None:
                result = call_method(*request.args, **request.kwargs)
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    result = await result
            elif callable(func):
                result = func(*request.args, **request.kwargs)
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    result = await result
            else:
                raise TypeError(f"Cannot call {request.function} on replica {self.idx}")

            if not request.future.done():
                request.future.set_result(result)
        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)
            self.mark_unhealthy()
            logger.error("Replica %d failed: %s", self.idx, e)
        finally:
            self.in_flight -= 1


class Router(abc.ABC):
    """Abstract base for request routing strategies."""

    @abc.abstractmethod
    def select(self, session_id: str | None, replicas: list[Replica]) -> Replica:
        """Pick a replica for the next request."""


class LeastLoadedRouter(Router):
    """Route to the replica with the fewest in-flight requests."""

    def __init__(self, max_concurrency: int = 16) -> None:
        self.max_concurrency = max_concurrency

    def select(self, session_id: str | None, replicas: list[Replica]) -> Replica:
        healthy = [r for r in replicas if r.is_healthy]
        if not healthy:
            raise RuntimeError("No healthy replicas available")
        best = min(healthy, key=lambda r: r.in_flight)
        if best.in_flight >= self.max_concurrency:
            logger.warning("All replicas at max capacity (%d)", self.max_concurrency)
        return best


class RoundRobinRouter(Router):
    """Simple round-robin routing across healthy replicas."""

    def __init__(self) -> None:
        self._counter = 0

    def select(self, session_id: str | None, replicas: list[Replica]) -> Replica:
        healthy = [r for r in replicas if r.is_healthy]
        if not healthy:
            raise RuntimeError("No healthy replicas available")
        idx = self._counter % len(healthy)
        self._counter += 1
        return healthy[idx]


class SessionRouter(Router):
    """Sticky routing: same session_id always goes to the same replica.

    Critical for agentic RL where KV cache locality matters.
    Falls back to least-loaded for new sessions.
    """

    def __init__(self) -> None:
        self._session_map: dict[str, int] = {}
        self._fallback = LeastLoadedRouter()

    def select(self, session_id: str | None, replicas: list[Replica]) -> Replica:
        if session_id is None:
            return self._fallback.select(None, replicas)

        if session_id in self._session_map:
            idx = self._session_map[session_id]
            if idx < len(replicas) and replicas[idx].is_healthy:
                return replicas[idx]
            del self._session_map[session_id]

        chosen = self._fallback.select(None, replicas)
        self._session_map[session_id] = chosen.idx
        return chosen


_ROUTER_REGISTRY: dict[str, type[Router]] = {
    "least_loaded": LeastLoadedRouter,
    "round_robin": RoundRobinRouter,
    "session": SessionRouter,
}


class Service:
    """Manages a pool of actor replicas with routing and health monitoring.

    Usage::

        service = Service(actor_refs, router="least_loaded")
        result = await service.call("generate", prompt, params)
        results = await service.broadcast("update_weights", version=3)
    """

    def __init__(
        self,
        actor_refs: list[ActorRef],
        router: str | Router = "least_loaded",
    ) -> None:
        self._replicas = [Replica(i, ref) for i, ref in enumerate(actor_refs)]
        if isinstance(router, str):
            router_cls = _ROUTER_REGISTRY.get(router, LeastLoadedRouter)
            self._router = router_cls()
        else:
            self._router = router
        logger.info(
            "Service started: %d replicas, router=%s",
            len(self._replicas),
            type(self._router).__name__,
        )

    @property
    def replicas(self) -> list[Replica]:
        return self._replicas

    async def call(
        self,
        function: str,
        *args: Any,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route a single request to one replica."""
        request = ServiceRequest(
            session_id=session_id,
            function=function,
            args=args,
            kwargs=kwargs,
            future=asyncio.get_running_loop().create_future(),
        )
        replica = self._router.select(session_id, self._replicas)
        asyncio.create_task(replica.dispatch(request))
        return await request.future

    async def broadcast(self, function: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a function on ALL healthy replicas and gather results."""
        healthy = [r for r in self._replicas if r.is_healthy]
        if not healthy:
            raise RuntimeError("No healthy replicas for broadcast")

        futures = []
        for replica in healthy:
            req = ServiceRequest(
                session_id=None,
                function=function,
                args=args,
                kwargs=kwargs,
                future=asyncio.get_running_loop().create_future(),
            )
            asyncio.create_task(replica.dispatch(req))
            futures.append(req.future)

        results = await asyncio.gather(*futures, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise RuntimeError(
                f"Broadcast failed on {len(errors)} replicas"
            ) from errors[0]
        return list(results)
