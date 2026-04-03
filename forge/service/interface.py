"""ServiceInterface -- user-facing API for interacting with distributed services.

Provides dynamic endpoint dispatch via ``ServiceEndpoint`` which supports:
- ``.route(*args)`` -- route to a single replica (load-balanced)
- ``.fanout(*args)`` -- broadcast to all healthy replicas
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import Generic, ParamSpec, TypeVar

from monarch._src.actor.endpoint import EndpointProperty

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_session_context = contextvars.ContextVar("session_context")


@dataclass
class Session:
    session_id: str


class SessionContext:
    """Async context manager for stateful service sessions.

    All requests within the context are routed to the same replica::

        async with service.session() as sess:
            r1 = await service.generate.route(prompt1)
            r2 = await service.generate.route(prompt2)  # same replica
    """

    def __init__(self, service: ServiceInterface):
        self.service = service
        self.session_id: str | None = None
        self._token = None

    async def __aenter__(self):
        self.session_id = await self.service.start_session()
        self._token = _session_context.set({"session_id": self.session_id})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            _session_context.reset(self._token)
        if self.session_id:
            await self.service.terminate_session(self.session_id)
            self.session_id = None


class ServiceEndpoint(Generic[P, R]):
    """Endpoint proxy for services.

    Replaces Monarch's native actor endpoint API with service-aware
    routing: ``.route()`` for single-replica and ``.fanout()`` for broadcast.
    """

    def __init__(self, service, endpoint_name: str):
        self.service = service
        self.endpoint_name = endpoint_name

    async def route(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Route request to one replica based on load balancing / session."""
        sess_id = kwargs.pop("sess_id", None)
        return await self.service._call(sess_id, self.endpoint_name, *args, **kwargs)

    async def fanout(self, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """Broadcast request to all healthy replicas."""
        return await self.service.call_all(self.endpoint_name, *args, **kwargs)

    async def call(self, *args: P.args, **kwargs: P.kwargs):
        raise NotImplementedError(
            "Services use .route() (single) or .fanout() (broadcast), not .call(). "
        )

    async def call_one(self, *args: P.args, **kwargs: P.kwargs):
        raise NotImplementedError(
            "Services use .route() (single) or .fanout() (broadcast), not .call_one(). "
        )


class ServiceInterface:
    """Lightweight handle to a Service, returned to callers.

    Dynamically creates ``ServiceEndpoint`` objects for each ``@endpoint``
    on the actor class, providing ``.route()`` and ``.fanout()`` APIs.
    """

    def __init__(self, _service, actor_def):
        self._service = _service
        self.actor_def = actor_def

        for attr_name in dir(actor_def):
            attr_value = getattr(actor_def, attr_name)
            if isinstance(attr_value, EndpointProperty):
                ep = ServiceEndpoint(self._service, attr_name)
                setattr(self, attr_name, ep)

    async def start_session(self) -> str:
        return await self._service.start_session()

    async def terminate_session(self, sess_id: str):
        return await self._service.terminate_session(sess_id)

    async def shutdown(self) -> None:
        await self._service.stop()

    def session(self) -> SessionContext:
        return SessionContext(self)

    async def get_metrics(self):
        return self._service.get_metrics()

    async def get_metrics_summary(self):
        return self._service.get_metrics_summary()

    def __getattr__(self, name: str):
        _service = object.__getattribute__(self, "_service")
        if hasattr(_service, name):
            return getattr(_service, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
