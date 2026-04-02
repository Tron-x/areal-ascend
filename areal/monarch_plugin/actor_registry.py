"""Actor lifecycle manager for Monarch orchestration.

Takes a list of :class:`MonarchActor` **classes** (not instances) and handles:

1. **Topological sort** on dependency graph (from ``cls.dependencies``)
2. **Ordered spawning** -- calls ``cls.spawn(ctx, host)`` in dependency order
3. **Ordered initialization** -- calls ``cls.initialize_actor(ctx)``
4. **Reverse-ordered shutdown** -- calls ``cls.shutdown_actor(ctx)``

Usage::

    from areal.monarch_plugin.generator_actor import GeneratorActor
    from areal.monarch_plugin.reward_actor import RewardActor
    ...

    actor_classes = [GeneratorActor, RewardActor, ...]
    registry = ActorRegistry(actor_classes)
    await registry.spawn_all(ctx, host)
    await registry.initialize_all(ctx)
    # ... run pipeline ...
    await registry.shutdown_all(ctx)
"""

from __future__ import annotations

from typing import Any

from areal.monarch_plugin.actor_spec import ActorContext
from areal.utils import logging

logger = logging.getLogger("ActorRegistry")


class ActorRegistry:
    """Manages the full lifecycle of MonarchActor classes."""

    def __init__(self, actor_classes: list[type]):
        self._classes: dict[str, type] = {
            cls.actor_name(): cls for cls in actor_classes
        }
        self._spawn_order: list[str] = []

        # Validate no duplicate names
        if len(self._classes) != len(actor_classes):
            names = [cls.actor_name() for cls in actor_classes]
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate actor names: {dupes}")

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _topo_sort(self) -> list[str]:
        """Kahn's algorithm: return names in spawn order."""
        in_degree: dict[str, int] = {n: 0 for n in self._classes}
        graph: dict[str, list[str]] = {n: [] for n in self._classes}

        for name, cls in self._classes.items():
            for dep in cls.dependencies:
                if dep not in self._classes:
                    raise ValueError(
                        f"Actor '{name}' depends on '{dep}' which is not in registry"
                    )
                graph[dep].append(name)
                in_degree[name] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbour in graph[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(self._classes):
            remaining = set(self._classes) - set(order)
            raise ValueError(f"Circular dependency detected among actors: {remaining}")
        return order

    # ------------------------------------------------------------------
    # Multi-replica generator support
    # ------------------------------------------------------------------

    @staticmethod
    def _is_multi_replica(ctx: ActorContext) -> bool:
        """Return True if generator should be spawned as multi-replica."""
        placements = ctx.extra.get("replica_placements")
        return placements is not None and len(placements) > 1

    async def _spawn_generator_replicas(self, ctx: ActorContext, host) -> None:
        """Spawn N GeneratorActor replicas and wrap them in a GeneratorService."""
        from areal.monarch_plugin.actor_base import MonarchActor, _per_host
        from areal.monarch_plugin.generator_actor import GeneratorActor
        from areal.monarch_plugin.generator_service import GeneratorService

        cls = GeneratorActor
        placements: list = ctx.extra["replica_placements"]

        replica_refs: dict[str, Any] = {}

        for rp in placements:
            name = rp.actor_name  # "generator_0", "generator_1", ...

            # 1. Per-replica ProcMesh with subset of device IDs
            bootstrap = cls.bootstrap_factory_for_replica(ctx, rp)
            resource = cls.resolve_resource(ctx)
            nprocs = cls.resolve_nprocs(ctx)
            per_host = _per_host(resource, nprocs)

            logger.info(
                f"Spawning ProcMesh for replica '{name}' (devices={rp.device_ids})"
            )
            procs = host.spawn_procs(per_host=per_host, bootstrap=bootstrap, name=name)
            ctx.procs[name] = procs

            # 2. Per-replica post_spawn (WorkerRegistry with unique name)
            extras = cls.post_spawn_for_replica(procs, name, ctx)
            if extras:
                ctx.extra_actors.update(extras)

            # 3. Per-replica constructor args (vLLM CLI with per-replica DP)
            ctor_kwargs = cls.constructor_args_for_replica(ctx, rp)
            resolved = MonarchActor._resolve_refs(ctor_kwargs, ctx)

            logger.info(f"Spawning replica actor '{name}' ({cls.__name__})")
            actor_ref = procs.spawn(name, cls, **resolved)
            ctx.actors[name] = actor_ref
            replica_refs[name] = actor_ref

        # 4. Replace individual refs with GeneratorService proxy
        per_replica_workers = ctx.alloc_mode.gen.tp_size * ctx.alloc_mode.gen.pp_size
        service = GeneratorService(
            replica_refs, placements, per_replica_workers=per_replica_workers
        )
        ctx.actors["generator"] = service

        logger.info(
            f"Created GeneratorService with {len(replica_refs)} replicas: "
            f"{list(replica_refs.keys())}"
        )

    async def _initialize_generator_replicas(self, ctx: ActorContext) -> Any:
        """Initialize all generator replicas and return aggregated result."""
        from areal.monarch_plugin.actor_base import MonarchActor
        from areal.monarch_plugin.generator_actor import GeneratorActor

        cls = GeneratorActor
        placements = ctx.extra["replica_placements"]
        results: dict[str, Any] = {}

        for rp in placements:
            name = rp.actor_name
            actor_ref = ctx.actors[name]
            method_name = cls.init_method()
            if method_name is None:
                continue

            init_kwargs = cls.init_args_for_replica(ctx, rp)
            resolved = MonarchActor._resolve_refs(init_kwargs, ctx)
            method = getattr(actor_ref, method_name)

            logger.info(f"Initialising replica '{name}' via {method_name}()")
            if resolved:
                result = await method.call_one(**resolved)
            else:
                result = await method.call_one()

            logger.info(f"  '{name}' initialised: {result}")
            results[name] = result

        # Return the first replica's result for compatibility
        return results.get(placements[0].actor_name, {})

    async def _shutdown_generator_replicas(self, ctx: ActorContext) -> None:
        """Shutdown all generator replicas in reverse order."""
        placements = ctx.extra.get("replica_placements", [])

        # Shutdown extra_actors for replica WorkerRegistries
        replica_extra_names = [
            f"worker_registry_{rp.replica_index}" for rp in placements
        ]
        for name in replica_extra_names:
            actor_ref = ctx.extra_actors.get(name)
            if actor_ref is not None:
                try:
                    await actor_ref.shutdown.call_one()
                except Exception as e:
                    logger.warning(f"Error shutting down extra actor '{name}': {e}")

        # Shutdown replicas in reverse order
        for rp in reversed(placements):
            name = rp.actor_name
            actor_ref = ctx.actors.get(name)
            procs = ctx.procs.get(name)

            if actor_ref is not None:
                try:
                    await actor_ref.shutdown.call_one()
                except Exception as e:
                    logger.warning(f"Error shutting down replica '{name}': {e}")

            if procs is not None:
                try:
                    procs.stop().get()
                except Exception as e:
                    logger.warning(f"Error stopping ProcMesh '{name}': {e}")

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    async def spawn_all(self, ctx: ActorContext, host) -> dict[str, Any]:
        """Spawn all actors in dependency order.

        Returns a dict mapping actor name -> ActorRef.
        """
        order = self._topo_sort()
        self._spawn_order = order
        logger.info(f"Actor spawn order: {order}")

        multi_replica = self._is_multi_replica(ctx)

        for name in order:
            if name == "generator" and multi_replica:
                await self._spawn_generator_replicas(ctx, host)
            else:
                cls = self._classes[name]
                await cls.spawn(ctx, host)

        return {**ctx.actors, **ctx.extra_actors}

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize_all(self, ctx: ActorContext) -> dict[str, Any]:
        """Call init_method on each actor (in dependency order).

        Returns a dict mapping actor name -> init result.
        """
        if not self._spawn_order:
            self._spawn_order = self._topo_sort()

        multi_replica = self._is_multi_replica(ctx)
        results: dict[str, Any] = {}

        for name in self._spawn_order:
            if name == "generator" and multi_replica:
                result = await self._initialize_generator_replicas(ctx)
                results["generator"] = result
            else:
                cls = self._classes[name]
                result = await cls.initialize_actor(ctx)
                results[name] = result

        return results

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown_all(self, ctx: ActorContext) -> None:
        """Shut down all actors and stop ProcMeshes in reverse spawn order."""
        order = list(reversed(self._spawn_order))
        logger.info(f"Shutdown order: {order}")

        multi_replica = self._is_multi_replica(ctx)

        # Shutdown extra actors first (e.g. worker_registry)
        for name, actor_ref in ctx.extra_actors.items():
            try:
                await actor_ref.shutdown.call_one()
            except Exception as e:
                logger.warning(f"Error shutting down extra actor '{name}': {e}")

        # Shutdown main actors in reverse order
        for name in order:
            if name == "generator" and multi_replica:
                await self._shutdown_generator_replicas(ctx)
            else:
                cls = self._classes[name]
                try:
                    await cls.shutdown_actor(ctx)
                except Exception as e:
                    logger.warning(f"Error during shutdown of '{name}': {e}")

        logger.info("All actors shut down.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def actor(self, name: str, ctx: ActorContext) -> Any:
        """Get actor reference by name from context."""
        all_actors = {**ctx.actors, **ctx.extra_actors}
        if name in all_actors:
            return all_actors[name]
        raise KeyError(f"No actor named '{name}'")

    @property
    def actor_classes(self) -> dict[str, type]:
        return dict(self._classes)
