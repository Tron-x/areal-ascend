"""``WeightSyncActor`` — actor-ify the weight-sync coordinator.

Wraps :class:`forge.engines.weight_sync.service.WeightSyncService` (the
4-beat trainer↔generator weight-sync orchestrator) inside a Monarch
``Actor`` so its lifecycle becomes Monarch-visible:

    * driver no longer holds a long-lived service object in its asyncio
      loop -- the coordinator runs in its own proc, dies cleanly when
      the proc mesh is torn down, shows up in Monarch's process list,
      and can be restarted independently of the algorithm front-end;
    * any new Adapter-Path framework can talk to the same actor instead
      of re-rolling its own coordinator (per
      ``framework-first-principles`` 准则 3: "any adapter must plug into
      Monarch -- no non-Monarch back doors").

The data plane (state-dict gather on trainer ranks, in-place RDMA /
broadcast-recv on generator workers) is unchanged: it still happens
inside :class:`~forge.actors.trainer.TrainerActor` and
:class:`~forge.actors.generator.WorkerWrapper`.  This actor only owns
the *control* plane: backend construction, planning, the push→pull
sequencing, and teardown.

**Conforms to:** none of the formal protocols in ``forge.core.protocols``
yet -- this is the first weight-sync actor in the codebase, so the
shape here will become the protocol when a second implementation lands
(``CrossMeshGroupActor`` for non-AReaL adapters).

Driver-side usage::

    handle = await spawn_weight_sync_actor(
        backend_name="torchstore_multi_vol",
        backend_kwargs={"pool_mb": 8192, "storage_mesh": storage_mesh},
        layout=layout,
        trainer=trainer_actor,
        generator=generator_actor,
        config={"forge_cfg": forge_cfg},
    )
    # Driver loop is unchanged: handle quacks like a legacy
    # WeightSyncStrategy (``.push(version)`` + ``.shutdown()``).
    await handle.push(version)
    ...
    await handle.shutdown()
"""

from __future__ import annotations

import logging
from typing import Any

from monarch.actor import Actor, endpoint, this_host

logger = logging.getLogger(__name__)


class WeightSyncActor(Actor):
    """Singleton coordinator that owns one ``WeightSyncService`` instance.

    Lives in its own 1-proc, no-GPU mesh; spawned on the driver host by
    default (overridable via ``host_mesh`` kwarg in
    :func:`spawn_weight_sync_actor`).
    """

    def __init__(self) -> None:
        super().__init__()
        # Late-bound by ``setup``; held here so subsequent ``push`` /
        # ``shutdown`` endpoints can reach the same service instance.
        self._service = None  # type: Any | None
        self._initialized = False

    @endpoint
    async def setup(
        self,
        *,
        backend_name: str,
        backend_kwargs: dict[str, Any],
        layout: Any,
        trainer: Any,
        generator: Any,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build backend + service, then call ``service.initialize()``.

        Returns whatever the backend's ``initialize`` returned (a meta
        dict the driver may want to log).  Raises on failure -- the
        driver is expected to catch and decide whether to teardown the
        actor or continue without sync (today's ``grpo.py`` does the
        latter).
        """
        # Lazy imports inside the endpoint so importing this module is
        # cheap on the driver, and so import errors surface where the
        # actor actually runs (not at module-load time on the driver).
        from forge.engines.weight_sync.backends import create_backend
        from forge.engines.weight_sync.service import WeightSyncService

        if self._initialized:
            raise RuntimeError(
                "WeightSyncActor.setup called twice on the same instance"
            )

        backend = create_backend(backend_name, **backend_kwargs)
        self._service = WeightSyncService(
            backend=backend,
            trainer_actor=trainer,
            generator_actor=generator,
            layout=layout,
            config=config or {},
        )
        meta = await self._service.initialize()
        self._initialized = True
        logger.info(
            "WeightSyncActor: initialized backend=%s, layout train=%d gen=%d tp=%d",
            backend_name,
            getattr(layout, "train_world", -1),
            getattr(layout, "gen_world", -1),
            getattr(layout, "gen_tp", -1),
        )
        return dict(meta) if isinstance(meta, dict) else {"meta": str(meta)}

    @endpoint
    async def push(self, version: int) -> dict[str, Any]:
        """Run one full sync cycle; returns the JSON-friendly summary.

        Identical contract to ``WeightSyncService.push(version)`` (the
        legacy ``WeightSyncStrategy`` shim) -- keeps the driver loop
        unchanged via :class:`WeightSyncActorHandle`.
        """
        if not self._initialized or self._service is None:
            raise RuntimeError("WeightSyncActor.push: setup not yet called")
        return await self._service.push(version)

    @endpoint
    async def get_status(self) -> dict[str, Any]:
        """Cheap health/version probe; never raises."""
        if self._service is None:
            return {"initialized": False}
        return await self._service.get_status()

    @endpoint
    async def shutdown(self) -> None:
        """Release backend resources (storage volumes / process groups)."""
        if self._service is None:
            return
        try:
            await self._service.shutdown()
        finally:
            self._service = None
            self._initialized = False


class WeightSyncActorHandle:
    """Thin async-method facade around a spawned ``WeightSyncActor``.

    Exposes the legacy ``WeightSyncStrategy`` interface (``push`` /
    ``shutdown``) so ``forge/apps/grpo.py``'s training loop doesn't need
    to learn about ``call_one`` semantics.  Also owns the proc mesh so
    a single ``await handle.shutdown()`` tears the whole actor down.
    """

    def __init__(self, actor: Any, proc_mesh: Any) -> None:
        self._actor = actor
        self._proc_mesh = proc_mesh

    @property
    def actor(self) -> Any:
        """Escape hatch for callers that need to invoke other endpoints."""
        return self._actor

    async def push(self, version: int) -> dict[str, Any]:
        return await self._actor.push.call_one(version)

    async def get_status(self) -> dict[str, Any]:
        return await self._actor.get_status.call_one()

    async def shutdown(self) -> None:
        # Two-step teardown: first ask the actor to release backend
        # resources gracefully, then tear down its proc mesh.  Swallow
        # endpoint errors (e.g. actor already dead) so mesh cleanup
        # still runs.
        try:
            await self._actor.shutdown.call_one()
        except Exception as e:
            logger.warning(
                "WeightSyncActorHandle: graceful shutdown raised %s: %s; "
                "proceeding with mesh teardown",
                type(e).__name__,
                e,
            )
        from forge.provisioner import stop_proc_mesh

        try:
            await stop_proc_mesh(self._proc_mesh)
        except Exception as e:
            logger.warning(
                "WeightSyncActorHandle: stop_proc_mesh raised %s: %s",
                type(e).__name__,
                e,
            )


async def spawn_weight_sync_actor(
    *,
    backend_name: str,
    backend_kwargs: dict[str, Any],
    layout: Any,
    trainer: Any,
    generator: Any,
    config: dict[str, Any] | None = None,
    host_mesh: Any | None = None,
) -> WeightSyncActorHandle:
    """Spawn a ``WeightSyncActor`` on a 1-proc / no-GPU mesh and initialize.

    Args:
        backend_name: backend factory key
            (``"torchstore_multi_vol"`` / ``"collective_broadcast"`` / ...).
        backend_kwargs: kwargs forwarded to ``create_backend``.  Any
            already-spawned ``ProcMesh`` (e.g. ``storage_mesh``) is
            passed through verbatim.
        layout: ``ParallelLayout`` describing trainer / generator world
            shapes.  Must be picklable (it's a plain ``@dataclass``).
        trainer: ``TrainerActor`` mesh ref (or any object the backend's
            ``push`` knows how to drive).
        generator: ``Generator`` (or its rollout-server analog) the
            backend's ``pull`` will drive.
        config: optional ``dict`` forwarded to ``backend.initialize`` --
            today only ``{"forge_cfg": forge_cfg}`` is used by the
            torchstore backends.
        host_mesh: where to place the actor's proc.  ``None`` (default)
            means ``this_host()`` -- the driver's own host, which is
            fine because the actor is control-plane only and does no
            heavy compute.  Pass a remote ``HostMesh`` when you want to
            colocate the coordinator with the trainer mesh on a remote
            machine.

    Returns:
        :class:`WeightSyncActorHandle` -- already past ``setup``, ready
        for ``await handle.push(version)``.
    """
    hm = host_mesh if host_mesh is not None else this_host()
    proc_mesh = hm.spawn_procs(per_host={"procs": 1})
    actor = proc_mesh.spawn("weight_sync", WeightSyncActor)
    try:
        await actor.setup.call_one(
            backend_name=backend_name,
            backend_kwargs=backend_kwargs,
            layout=layout,
            trainer=trainer,
            generator=generator,
            config=config,
        )
    except Exception:
        # Setup failed -- tear the proc mesh down so we don't leak a
        # dangling actor proc on init failure.
        from forge.provisioner import stop_proc_mesh

        try:
            await stop_proc_mesh(proc_mesh)
        except Exception:
            pass
        raise
    return WeightSyncActorHandle(actor, proc_mesh)
