"""MonarchActor base class — declarative actor with .spawn() lifecycle.

Each concrete actor inherits from :class:`MonarchActor` and declares its
resource requirements, dependencies, and lifecycle hooks as class-level
attributes and classmethod overrides.  The base class provides:

- ``cls.spawn(ctx, host)`` — create ProcMesh, run post_spawn, spawn actor
- ``cls.initialize_actor(ctx)`` — call the init endpoint with resolved args
- ``cls.shutdown_actor(ctx)`` — graceful teardown
- ``cls._resolve_refs(args, ctx)`` — resolve ``ActorRef`` / ``CtxRef`` markers

The :class:`ActorRegistry` calls these in topological order.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from monarch.actor import Actor

from areal.monarch_plugin.actor_spec import ActorContext, ActorRef, CtxRef, ResourceKind
from areal.utils import logging

logger = logging.getLogger("MonarchActor")


def _per_host(resource: ResourceKind, nprocs: int) -> dict:
    """Build ``per_host`` dict for ``host.spawn_procs()``."""
    if resource == ResourceKind.CPU:
        return {"cpu": 1}
    elif resource == ResourceKind.NPU_SINGLE:
        return {"npu": 1}
    elif resource == ResourceKind.NPU_MULTI:
        return {"npu": nprocs}
    raise ValueError(f"Unknown ResourceKind: {resource}")


def _to_snake(name: str) -> str:
    """Convert CamelCase class name to snake_case actor name."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    # Remove trailing "Actor" suffix for cleaner names
    s = re.sub(r"_[Aa]ctor$", "", s)
    return s.lower()


class MonarchActor(Actor):
    """Base class for Monarch actors with declarative resource/dependency metadata.

    Subclasses override class-level attributes and classmethods to declare:
      - ``resource`` — CPU, NPU_SINGLE, or NPU_MULTI
      - ``dependencies`` — list of actor names that must spawn first
      - ``bootstrap_factory(ctx)`` — ProcMesh bootstrap function
      - ``constructor_args(ctx)`` — kwargs for ``__init__``
      - ``init_method()`` — name of endpoint to call after spawn (or None)
      - ``init_args(ctx)`` — kwargs for the init endpoint
      - ``post_spawn(procs, ctx)`` — hook for extra actors on same ProcMesh

    For dynamic resources (e.g. TrainerActor where nprocs depends on config),
    override ``resolve_resource(ctx)`` and ``resolve_nprocs(ctx)``.
    """

    # --- Class-level declarations (override in subclasses) ---
    resource: ResourceKind = ResourceKind.CPU
    nprocs: int = 1
    dependencies: list[str] = []
    is_multi_rank: bool = False
    shutdown_broadcast: bool = False

    # --- Naming ---

    @classmethod
    def actor_name(cls) -> str:
        """Unique name for this actor type (derived from class name)."""
        return _to_snake(cls.__name__)

    # --- Dynamic resource resolution (override for dynamic actors) ---

    @classmethod
    def resolve_resource(cls, ctx: ActorContext) -> ResourceKind:
        """Return the ResourceKind for this actor. Override for dynamic logic."""
        return cls.resource

    @classmethod
    def resolve_nprocs(cls, ctx: ActorContext) -> int:
        """Return the number of processes. Override for dynamic logic."""
        return cls.nprocs

    @classmethod
    def resolve_is_multi_rank(cls, ctx: ActorContext) -> bool:
        return cls.resolve_nprocs(ctx) > 1

    @classmethod
    def resolve_shutdown_broadcast(cls, ctx: ActorContext) -> bool:
        """Whether to use call() for shutdown instead of call_one()."""
        return cls.shutdown_broadcast

    # --- Lifecycle classmethods (override in subclasses) ---

    @classmethod
    def bootstrap_factory(cls, ctx: ActorContext) -> Callable:
        """Return a bootstrap callable for ProcMesh creation.

        Default: CPU bootstrap (no-op).
        """
        from areal.monarch_plugin.bootstraps import make_cpu_bootstrap

        return make_cpu_bootstrap

    @classmethod
    def constructor_args(cls, ctx: ActorContext) -> dict:
        """Build constructor kwargs. Override to declare ActorRef/CtxRef."""
        return {}

    @classmethod
    def init_method(cls) -> str | None:
        """If set, this method is called on the actor after spawning."""
        return None

    @classmethod
    def init_args(cls, ctx: ActorContext) -> dict:
        """Build init kwargs. Override to pass ActorRef/CtxRef references."""
        return {}

    @classmethod
    def post_spawn(cls, procs: Any, ctx: ActorContext) -> dict[str, Any]:
        """Hook called after ProcMesh creation but before actor init.

        Returns a dict of extra actor references to merge into context.
        """
        return {}

    # --- Reference resolution ---

    @staticmethod
    def _resolve_refs(args: dict, ctx: ActorContext) -> dict:
        """Walk *args* and resolve ActorRef / CtxRef markers."""
        resolved: dict[str, Any] = {}
        all_actors = {**ctx.actors, **ctx.extra_actors}
        for k, v in args.items():
            if isinstance(v, ActorRef):
                if v.name not in all_actors:
                    raise KeyError(
                        f"ActorRef('{v.name}') not yet spawned. "
                        f"Available: {list(all_actors)}"
                    )
                resolved[k] = all_actors[v.name]
            elif isinstance(v, CtxRef):
                resolved[k] = ctx.extra[v.key]
            else:
                resolved[k] = v
        return resolved

    # --- Spawn ---

    @classmethod
    async def spawn(cls, ctx: ActorContext, host) -> Any:
        """Create ProcMesh, run post_spawn hook, and spawn this actor.

        Returns the actor reference.
        """
        resource = cls.resolve_resource(ctx)
        nprocs = cls.resolve_nprocs(ctx)
        name = cls.actor_name()

        bootstrap = cls.bootstrap_factory(ctx)
        per_host = _per_host(resource, nprocs)

        logger.info(
            f"Spawning ProcMesh for '{name}' (resource={resource.value}, nprocs={nprocs})"
        )
        procs = host.spawn_procs(per_host=per_host, bootstrap=bootstrap, name=name)
        ctx.procs[name] = procs

        # post_spawn hook
        extras = cls.post_spawn(procs, ctx)
        if extras:
            ctx.extra_actors.update(extras)
            logger.info(f"  post_spawn added actors: {list(extras.keys())}")

        # Resolve constructor args and spawn the actor
        ctor_kwargs = cls._resolve_refs(cls.constructor_args(ctx), ctx)

        logger.info(f"Spawning actor '{name}' ({cls.__name__})")
        actor_ref = procs.spawn(name, cls, **ctor_kwargs)
        ctx.actors[name] = actor_ref
        return actor_ref

    # --- Initialize ---

    @classmethod
    async def initialize_actor(cls, ctx: ActorContext) -> Any:
        """Call the init endpoint on the spawned actor, if init_method is set."""
        method_name = cls.init_method()
        if method_name is None:
            return None

        name = cls.actor_name()
        actor_ref = ctx.actors[name]
        multi_rank = cls.resolve_is_multi_rank(ctx)
        method = getattr(actor_ref, method_name)

        logger.info(f"Initialising '{name}' via {method_name}()")
        if multi_rank:
            result_mesh = await method.call()
            result = (
                result_mesh.item(npu=0)
                if hasattr(result_mesh, "item")
                else result_mesh
            )
        else:
            init_kwargs = cls._resolve_refs(cls.init_args(ctx), ctx)
            if init_kwargs:
                result = await method.call_one(**init_kwargs)
            else:
                result = await method.call_one()

        logger.info(f"  '{name}' initialised: {result}")
        return result

    # --- Shutdown ---

    @classmethod
    async def shutdown_actor(cls, ctx: ActorContext) -> None:
        """Shutdown the actor and stop the ProcMesh."""
        name = cls.actor_name()
        actor_ref = ctx.actors.get(name)
        procs = ctx.procs.get(name)

        if actor_ref is not None:
            try:
                if cls.resolve_shutdown_broadcast(ctx):
                    await actor_ref.shutdown.call()
                else:
                    await actor_ref.shutdown.call_one()
            except Exception as e:
                logger.warning(f"Error shutting down actor '{name}': {e}")

        if procs is not None:
            try:
                procs.stop().get()
            except Exception as e:
                logger.warning(f"Error stopping ProcMesh '{name}': {e}")
