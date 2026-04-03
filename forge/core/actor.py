"""Abstract ForgeActor base class.

This is the framework-agnostic actor abstraction. Backend-specific
implementations (``MonarchForgeActor``, ``RayForgeActor``) inherit from
this and provide concrete spawning logic.

Unlike TorchForge's ``ForgeActor`` (which inherits ``monarch.actor.Actor``
directly), this class is a pure ABC -- no Monarch or Ray dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T", bound="ForgeActor")


@dataclass
class ActorConfig:
    """Configuration snapshot returned by ``ForgeActor.options()``.

    Carries resource overrides and the target class, to be consumed
    by ``as_actor()`` or ``as_service()`` on the appropriate backend.
    """

    cls: type
    procs: int = 1
    with_gpus: bool = False
    num_replicas: int = 1
    hosts: int | None = None
    bootstrap: Callable | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    async def as_actor(self, *args: Any, **kwargs: Any) -> Any:
        """Spawn a single actor instance via the configured backend.

        This delegates to ``cls._spawn_actor(self, *args, **kwargs)``.
        """
        return await self.cls._spawn_actor(self, *args, **kwargs)

    async def as_service(self, *args: Any, **kwargs: Any) -> Any:
        """Spawn a multi-replica service via the configured backend.

        This delegates to ``cls._spawn_service(self, *args, **kwargs)``.
        """
        return await self.cls._spawn_service(self, *args, **kwargs)


class ForgeActor(ABC):
    """Abstract base class for all Forge actors.

    Subclasses declare resource requirements as class attributes and
    implement ``setup()`` / ``shutdown()`` lifecycle hooks.

    Resource declarations::

        class MyActor(ForgeActor):
            procs = 4
            with_gpus = True
            num_replicas = 2

    Spawning::

        actor = await MyActor.options(procs=8).as_actor(arg1, arg2)
        service = await MyActor.options(num_replicas=3).as_service()
    """

    procs: int = 1
    with_gpus: bool = False
    num_replicas: int = 1
    hosts: int | None = None

    @classmethod
    def options(
        cls: type[T],
        procs: int | None = None,
        with_gpus: bool | None = None,
        num_replicas: int | None = None,
        hosts: int | None = None,
        bootstrap: Callable | None = None,
        **extra: Any,
    ) -> ActorConfig:
        """Create an ``ActorConfig`` with optional resource overrides.

        Any parameter not provided falls back to the class attribute.
        """
        return ActorConfig(
            cls=cls,
            procs=procs if procs is not None else cls.procs,
            with_gpus=with_gpus if with_gpus is not None else cls.with_gpus,
            num_replicas=num_replicas if num_replicas is not None else cls.num_replicas,
            hosts=hosts if hosts is not None else cls.hosts,
            bootstrap=bootstrap,
            extra=extra,
        )

    @classmethod
    async def _spawn_actor(cls, config: ActorConfig, *args: Any, **kwargs: Any) -> Any:
        """Backend-specific actor spawn. Override in adapter subclasses."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement _spawn_actor. "
            "Use a backend-specific subclass (e.g. MonarchForgeActor)."
        )

    @classmethod
    async def _spawn_service(
        cls, config: ActorConfig, *args: Any, **kwargs: Any
    ) -> Any:
        """Backend-specific service spawn. Override in adapter subclasses."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement _spawn_service. "
            "Use a backend-specific subclass (e.g. MonarchForgeActor)."
        )

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the actor after spawning (load models, connect, etc.)."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources on teardown."""
