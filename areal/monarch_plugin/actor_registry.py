"""Actor lifecycle manager for Monarch orchestration.

Takes a list of :class:`ActorSpec` instances and handles:

1. **Topological sort** on dependency graph
2. **Ordered spawning** -- ProcMesh creation, post_spawn hooks, actor init
3. **Reverse-ordered shutdown** -- graceful cleanup

Usage::

    specs = [ActorSpec(...), ActorSpec(...)]
    registry = ActorRegistry(specs)
    actors = await registry.spawn_all(ctx)
    info = await registry.initialize_all(ctx)
    # ... run pipeline ...
    await registry.shutdown_all()
"""

from __future__ import annotations

from typing import Any

from areal.monarch_plugin.actor_spec import ActorContext, ActorRef, CtxRef, ResourceKind
from areal.utils import logging

logger = logging.getLogger("ActorRegistry")


class ActorRegistry:
    """Manages the full lifecycle of Monarch actors declared via specs."""

    def __init__(self, specs: list):
        self._specs: dict[str, Any] = {s.name: s for s in specs}
        self._actors: dict[str, Any] = {}
        self._procs: dict[str, Any] = {}
        self._extra_actors: dict[str, Any] = {}

        # Validate no duplicate names
        if len(self._specs) != len(specs):
            names = [s.name for s in specs]
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate actor spec names: {dupes}")

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _topo_sort(self) -> list[str]:
        """Kahn's algorithm: return names in spawn order."""
        in_degree: dict[str, int] = {n: 0 for n in self._specs}
        graph: dict[str, list[str]] = {n: [] for n in self._specs}

        for name, spec in self._specs.items():
            for dep in spec.dependencies:
                if dep not in self._specs:
                    raise ValueError(
                        f"Actor '{name}' depends on '{dep}' which is not in specs"
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

        if len(order) != len(self._specs):
            remaining = set(self._specs) - set(order)
            raise ValueError(f"Circular dependency detected among actors: {remaining}")
        return order

    # ------------------------------------------------------------------
    # ProcMesh helper
    # ------------------------------------------------------------------

    def _per_host(self, spec) -> dict:
        if spec.resource == ResourceKind.CPU:
            return {"cpu": 1}
        elif spec.resource == ResourceKind.NPU_SINGLE:
            return {"npu": 1}
        elif spec.resource == ResourceKind.NPU_MULTI:
            return {"npu": spec.nprocs}
        raise ValueError(f"Unknown ResourceKind: {spec.resource}")

    # ------------------------------------------------------------------
    # Reference resolution
    # ------------------------------------------------------------------

    def _resolve_refs(self, args: dict, ctx: ActorContext) -> dict:
        """Walk *args* and resolve :class:`ActorRef` / :class:`CtxRef`.

        Plain values are passed through unchanged.
        """
        resolved: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, ActorRef):
                resolved[k] = self.actor(v.name)
            elif isinstance(v, CtxRef):
                resolved[k] = ctx.extra[v.key]
            else:
                resolved[k] = v
        return resolved

    def _resolve_args(
        self,
        args: dict[str, Any] | Callable[[ActorContext], dict] | None,
        ctx: ActorContext,
    ) -> dict[str, Any]:
        """Resolve constructor/init args — supports dict or callable."""
        if args is None:
            return {}
        if isinstance(args, dict):
            return self._resolve_refs(args, ctx)
        return args(ctx)

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    async def spawn_all(self, ctx: ActorContext, host) -> dict[str, Any]:
        """Create ProcMeshes and spawn actors in dependency order.

        Returns a dict mapping actor name -> ActorRef.
        """
        order = self._topo_sort()
        logger.info(f"Actor spawn order: {order}")

        spawned: list[str] = []
        try:
            for name in order:
                spec = self._specs[name]
                ctx.actors = {**self._actors, **self._extra_actors}

                bootstrap = spec.bootstrap_factory()
                per_host = self._per_host(spec)

                logger.info(
                    f"Spawning ProcMesh for '{name}' "
                    f"(resource={spec.resource.value})"
                )
                procs = host.spawn_procs(
                    per_host=per_host,
                    bootstrap=bootstrap,
                    name=name,
                )
                self._procs[name] = procs
                spawned.append(name)

                # post_spawn hook (e.g. WorkerRegistry on generator ProcMesh)
                if spec.post_spawn is not None:
                    extras = spec.post_spawn(procs, ctx)
                    if extras:
                        self._extra_actors.update(extras)
                        ctx.actors = {**self._actors, **self._extra_actors}
                        logger.info(
                            f"  post_spawn added actors: {list(extras.keys())}"
                        )

                # Build constructor args (dict or callable)
                ctor_kwargs = self._resolve_args(spec.constructor_args, ctx)

                logger.info(
                    f"Spawning actor '{name}' ({spec.actor_class.__name__})"
                )
                actor_ref = procs.spawn(name, spec.actor_class, **ctor_kwargs)
                self._actors[name] = actor_ref

        except Exception:
            logger.error(
                f"Spawn failed at '{name}', "
                f"rolling back {len(spawned)} ProcMeshes"
            )
            for rollback_name in reversed(spawned):
                procs = self._procs.pop(rollback_name, None)
                if procs is not None:
                    try:
                        procs.stop().get()
                    except Exception:
                        pass
            raise

        return {**self._actors, **self._extra_actors}

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize_all(self, ctx: ActorContext) -> dict[str, Any]:
        """Call init_method on each actor (in dependency order).

        Returns a dict mapping actor name -> init result.
        """
        order = self._topo_sort()
        results: dict[str, Any] = {}

        for name in order:
            spec = self._specs[name]
            if spec.init_method is None:
                continue

            ctx.actors = {**self._actors, **self._extra_actors}
            actor_ref = self._actors[name]
            method = getattr(actor_ref, spec.init_method)

            logger.info(f"Initialising '{name}' via {spec.init_method}()")
            if spec.is_multi_rank:
                result_mesh = await method.call()
                result = (
                    result_mesh.item(npu=0)
                    if hasattr(result_mesh, "item")
                    else result_mesh
                )
            else:
                init_kwargs = self._resolve_args(spec.init_args, ctx)
                if init_kwargs:
                    result = await method.call_one(**init_kwargs)
                else:
                    result = await method.call_one()

            results[name] = result
            logger.info(f"  '{name}' initialised: {result}")

        return results

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown_all(self) -> None:
        """Shut down all actors and stop ProcMeshes in reverse spawn order."""
        order = list(reversed(list(self._procs.keys())))
        logger.info(f"Shutdown order: {order}")

        # Shutdown actors
        for name in order:
            spec = self._specs.get(name)
            actor_ref = self._actors.get(name)
            if actor_ref is None:
                continue
            try:
                if spec and spec.shutdown_broadcast:
                    await actor_ref.shutdown.call()
                else:
                    await actor_ref.shutdown.call_one()
            except Exception as e:
                logger.warning(f"Error shutting down actor '{name}': {e}")

        # Shutdown extra actors (e.g. worker_registry)
        for name, actor_ref in self._extra_actors.items():
            try:
                await actor_ref.shutdown.call_one()
            except Exception as e:
                logger.warning(f"Error shutting down extra actor '{name}': {e}")

        # Stop ProcMeshes
        for name in order:
            procs = self._procs.get(name)
            if procs is None:
                continue
            try:
                procs.stop().get()
            except Exception as e:
                logger.warning(f"Error stopping ProcMesh '{name}': {e}")

        logger.info("All actors shut down.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def actors(self) -> dict[str, Any]:
        return {**self._actors, **self._extra_actors}

    def actor(self, name: str) -> Any:
        """Get actor reference by name."""
        if name in self._actors:
            return self._actors[name]
        if name in self._extra_actors:
            return self._extra_actors[name]
        raise KeyError(f"No actor named '{name}'")
