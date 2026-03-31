"""Declarative actor specification for Monarch orchestration.

Instead of imperatively spawning each actor in a monolithic function,
each actor type is described by an :class:`ActorSpec` dataclass.  The
:class:`ActorRegistry` consumes these specs, resolves dependencies via
topological sort, and handles the full lifecycle automatically.

Example::

    specs = [
        ActorSpec(name="generator", actor_class=GeneratorActor, ...),
        ActorSpec(name="reward", actor_class=RewardActor, ...),
    ]
    registry = ActorRegistry(specs)
    actors = await registry.spawn_all(ctx)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceKind(Enum):
    """What kind of ProcMesh to create for an actor."""

    CPU = "cpu"  # per_host={"cpu": 1}
    NPU_SINGLE = "npu_single"  # per_host={"npu": 1}, single device visible
    NPU_MULTI = "npu_multi"  # per_host={"npu": N}, all training devices visible


@dataclass
class ActorContext:
    """Bundles everything a spec callback might need to construct args.

    Passed to ``constructor_args`` and ``init_args`` callables so they can
    reference config, topology, and already-spawned actors.
    """

    config: Any
    alloc_mode: Any
    placement: Any
    host: Any
    actors: dict[str, Any] = field(default_factory=dict)
    master_port: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActorSpec:
    """Declarative description of one Monarch actor.

    Parameters
    ----------
    name:
        Unique identifier used as key in the actors dict and for dependency
        resolution.
    actor_class:
        The Monarch ``Actor`` subclass to spawn.
    resource:
        Determines the ``per_host`` ProcMesh configuration.
    bootstrap_factory:
        ``() -> callable``  Returns the bootstrap function passed to
        ``host.spawn_procs(bootstrap=...)``.
    constructor_args:
        ``(ctx: ActorContext) -> dict``  Returns keyword arguments for the
        actor constructor.  ``ctx.actors`` contains previously spawned actors
        so dependencies can be wired.
    dependencies:
        Names of actors that must be spawned (and optionally initialised)
        before this one.
    init_method:
        If set, this method is called on the actor after spawning (e.g.
        ``"setup"`` or ``"initialize"``).
    init_args:
        ``(ctx: ActorContext) -> dict``  Arguments for the ``init_method`` call.
    post_spawn:
        ``(procs, ctx: ActorContext) -> dict[str, ActorRef]``  Called after the
        ProcMesh is created but before the main actor is initialised.  Use this
        to spawn additional actors on the same ProcMesh (e.g. WorkerRegistry).
        Returns a dict of extra actor references to merge into the registry.
    is_multi_rank:
        If ``True``, the ProcMesh has >1 process and ``call()`` (broadcast)
        should be used instead of ``call_one()``.
    shutdown_broadcast:
        If ``True``, use ``call()`` for shutdown instead of ``call_one()``.
        Needed when the actor has multiple ranks (e.g. TrainerActor).
    """

    name: str
    actor_class: type
    resource: ResourceKind
    bootstrap_factory: Callable[[], Callable]
    constructor_args: Callable[[ActorContext], dict]
    dependencies: list[str] = field(default_factory=list)
    init_method: str | None = None
    init_args: Callable[[ActorContext], dict] | None = None
    post_spawn: Callable[[Any, ActorContext], dict[str, Any]] | None = None
    nprocs: int = 1
    """Number of processes in the ProcMesh.  Only used when ``resource=NPU_MULTI``."""
    is_multi_rank: bool = False
    shutdown_broadcast: bool = False
