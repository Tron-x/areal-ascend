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
    # Spawn
    # ------------------------------------------------------------------

    async def spawn_all(self, ctx: ActorContext, host) -> dict[str, Any]:
        """Spawn all actors in dependency order.

        Returns a dict mapping actor name -> ActorRef.
        """
        order = self._topo_sort()
        self._spawn_order = order
        logger.info(f"Actor spawn order: {order}")

        spawned: list[str] = []
        for name in order:
            cls = self._classes[name]
            await cls.spawn(ctx, host)
            spawned.append(name)

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

        results: dict[str, Any] = {}
        for name in self._spawn_order:
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

        # Shutdown extra actors first (e.g. worker_registry)
        for name, actor_ref in ctx.extra_actors.items():
            try:
                await actor_ref.shutdown.call_one()
            except Exception as e:
                logger.warning(f"Error shutting down extra actor '{name}': {e}")

        # Shutdown main actors in reverse order
        for name in order:
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
