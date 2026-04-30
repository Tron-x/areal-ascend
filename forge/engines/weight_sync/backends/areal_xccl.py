"""AReaL-style xccl backend: trainer rank 0 broadcasts directly to vLLM workers.

Compared to :class:`CollectiveBroadcastBackend` -- which inserts a torchstore
volume as the bcast source so HiXL handles the put side -- this backend keeps
the path AReaL itself uses in production: trainer rank 0 builds the flat
buffer in-process and emits a single ``torch.distributed.broadcast`` directly
into the vLLM workers' receive group.  No torchstore, no HiXL, no extra hop;
purely standard c10d on top of either NCCL (CUDA) or HCCL (Ascend NPU).

Why keep this option even when multi-vol is faster:

* **Zero HiXL / RDMA dependency.**  The whole transport is whatever
  ``torch.distributed`` already has for the chosen device.  On a fresh
  CANN install with no HiXL stack, this still works.
* **Multi-host NPU correctness.**  ``init_custom_process_group`` goes
  through PyTorch's c10d-HCCL backend, which dispatches to the cluster-info
  flavour of HCCL init and supports cross-host rendezvous -- unlike
  ``vllm_ascend.PyHcclCommunicator`` which wraps the single-host
  ``HcclCommInitRootInfo`` API and refuses to set up a cross-host
  communicator (tracked separately).  This makes ``areal_xccl`` the
  natural fallback whenever vLLM's own collective comms misbehave.
* **Proven at scale.**  AReaL has production traces running this pattern
  (its ``FSDPEngine._update_weights_from_distributed`` path); the building
  blocks (``init_custom_process_group``, ``VLLMWorkerExtension``) live in
  :mod:`areal.weight_sync` and are reused verbatim.
* **Debugging baseline.**  When comparing HiXL-based backends to a
  known-good reference, flipping a config flag and seeing the same weights
  delivered via plain HCCL broadcast is priceless.

Rank layout (kept identical to AReaL's, so trainers are interoperable):

    xccl_world = 1 + gen.world_size
    rank 0     = trainer rank 0 (broadcast source)
    rank 1..N  = generator TP/PP workers, indexed
                 ``1 + server_idx * tp * pp + local_rank``

Other trainer ranks (1..train_world-1) participate in the FSDP gather inside
the trainer process but do **not** join this xccl group; only rank 0
broadcasts the gathered flat tensor.

Required endpoint contract
--------------------------

This backend reuses the generator-worker endpoints that already exist for
:class:`CollectiveBroadcastBackend` (and have been validated at TP=4 by
``forge/scripts/test_cross_mesh_bcast*.py``):

* ``generator.workers.init_bcast_group(master_addr, master_port, world_size,
  rank, backend, group_name, timeout_s)`` -- workers join the xccl group.
* ``generator.workers.recv_and_load_flat(version, plan, total_bytes,
  src_rank=0)`` -- workers receive a single flat buffer broadcast and unpack
  into model parameters.
* ``generator.workers.shutdown_bcast_group()`` -- workers tear down the
  group.

The backend additionally **expects** the trainer actor (whatever
``trainer_actor`` is passed into :meth:`initialize`) to expose three
sibling endpoints; until they exist, :meth:`initialize`/:meth:`push` will
raise an informative error.  Their suggested signatures, written so a
follow-up PR can land them in ``forge/actors/trainer.py`` next to
``publish_weights_flat``:

.. code-block:: python

    @endpoint
    def init_xccl_source(
        self,
        master_addr: str,
        master_port: int,
        world_size: int,
        rank: int = 0,
        backend: str = "hccl",
        group_name: str = "areal_xccl_source",
        timeout_s: int = 180,
    ) -> dict:
        '''Trainer rank 0 only: build the bcast-source ProcessGroup
        via ``areal.weight_sync.init_custom_process_group`` and stash
        it under ``self._xccl_groups[group_name]``.  Other trainer
        ranks return immediately.'''

    @endpoint
    def publish_weights_xccl(
        self,
        version: int,
        group_name: str = "areal_xccl_source",
    ) -> dict:
        '''All trainer ranks: FSDP-gather the state_dict (collective).
        Rank 0 then packs the gathered tensors into a single flat
        buffer and ``dist.broadcast(flat, src=0, group=...)`` it on
        the xccl group.  Returns ``{"plan": [...], "total_bytes": N,
        "build_state_dict_s": ..., "broadcast_s": ...}`` from rank 0;
        other ranks return ``{"rank": r}`` only.

        ``plan`` matches the layout produced by
        ``forge.engines.weight_sync._flat_layout.plan_layout`` so the
        worker's ``recv_and_load_flat`` can unpack it without
        renegotiation.'''

    @endpoint
    def shutdown_xccl_source(
        self,
        group_name: str = "areal_xccl_source",
    ) -> None:
        '''Trainer rank 0 only: ``dist.destroy_process_group`` for
        the named group and drop the entry from
        ``self._xccl_groups``.'''

Bandwidth note (vs ``torchstore_multi_vol`` and ``collective_broadcast``):

    multi_vol           put = N NICs parallel,  pull = M NICs parallel
    collective_broadcast put = 1 HiXL link,      pull = HCCL tree
    areal_xccl          put = 1 NIC out of trainer rank 0,
                        pull = HCCL tree (embedded in same broadcast)

So ``areal_xccl`` is bandwidth-bounded by trainer rank 0's NIC -- strictly
slower than multi_vol on the put side -- but it is the simplest path that
works without HiXL, and it is what we lean on whenever a vendor's HCCL
wrapper has a cross-host correctness issue we don't yet have time to chase
upstream.

Known limitations (mirrored from AReaL):

* vLLM LoRA via xccl: not yet supported here (AReaL's ``update_weight_lora_xccl``
  is wired but our flat-buffer path doesn't carry adapter metadata).
* ``gen.pp_size != 1``: not implemented (matches AReaL's SGLang xccl path).
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from typing import Any

from forge.engines.weight_sync.service import (
    ParallelLayout,
    PullResult,
    PushResult,
)

logger = logging.getLogger("ArealXcclBackend")


class ArealXcclBackend:
    """Trainer-rank-0 -> generator-workers HCCL broadcast, no storage hop."""

    def __init__(
        self,
        *,
        master_addr: str | None = None,
        master_port: int | None = None,
        bcast_backend: str = "hccl",
        bcast_timeout_s: int = 180,
        group_name: str = "areal_xccl_source",
    ) -> None:
        """
        Args:
            master_addr: Hostname/IP that vLLM workers connect to for the
                HCCL TCPStore rendezvous.  ``None`` resolves to the trainer
                host's hostname at :meth:`initialize` time.
            master_port: TCP port used by ``init_process_group`` for the
                rendezvous.  ``None`` picks a free port at initialize time.
            bcast_backend: ``torch.distributed`` backend string.  Default
                ``"hccl"`` for Ascend NPU; set ``"nccl"`` on CUDA builds.
            bcast_timeout_s: HCCL init / broadcast timeout.
            group_name: Name passed to ``init_custom_process_group`` /
                ``init_bcast_group`` so trainer + workers refer to the same
                PG.  Override only if multiple xccl groups coexist in the
                same process (advanced).
        """
        self._configured_master_addr = master_addr
        self._configured_master_port = master_port
        self._bcast_backend = bcast_backend
        self._bcast_timeout_s = bcast_timeout_s
        self._group_name = group_name

        self._layout: ParallelLayout | None = None
        self._trainer_actor: Any | None = None
        self._generator_actor: Any | None = None
        self._initialized = False

        # Stats from the most recent push, returned verbatim by pull so the
        # service control loop sees a consistent (push, pull) pair.  AReaL
        # xccl is a one-shot broadcast: workers receive inside the same
        # collective the trainer source emits, so there is no separate
        # bytes-on-wire phase to attribute to "pull".
        self._cached_pull_result: PullResult | None = None
        self._resolved_master_addr: str | None = None
        self._resolved_master_port: int | None = None

    # ------------------------------------------------------------------
    # Public Protocol surface
    # ------------------------------------------------------------------

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        layout: ParallelLayout,
        config: dict,
    ) -> dict:
        """Build the cross-mesh xccl ProcessGroup spanning trainer rank 0
        + every generator TP/PP worker.

        Steps:

        1. Resolve ``(master_addr, master_port)`` -- caller-supplied, or
           ``trainer.get_host()`` / hostname + free port pick.
        2. Concurrently fire the rendezvous on both legs:
           ``trainer.init_xccl_source.call_one(rank=0, ...)`` and
           ``generator.workers.init_bcast_group.call(rank=1+i, ...)``.
        3. Cache the resolved endpoint so :meth:`shutdown` can reach the
           same group.

        Caller responsibilities:

        * The trainer actor must expose ``init_xccl_source`` matching the
          contract in this module's docstring.  If it doesn't,
          ``initialize`` raises with a clear message.
        * The generator actor must expose ``workers`` (an ActorMesh) with
          ``init_bcast_group`` -- already true in
          ``forge/actors/generator/worker.py`` as of MVP-2.
        """
        if layout.gen_pp != 1:
            raise NotImplementedError(
                "ArealXcclBackend: gen_pp != 1 is not supported "
                f"(got gen_pp={layout.gen_pp}). Matches AReaL's SGLang "
                "xccl path; fix is to extend the rank scheme to include pp."
            )

        self._layout = layout
        self._trainer_actor = trainer_actor
        self._generator_actor = generator_actor

        # ---- Resolve rendezvous endpoint -------------------------------
        master_addr = self._configured_master_addr or await self._resolve_trainer_host(
            trainer_actor
        )
        master_port = self._configured_master_port or _find_free_port()

        tp = max(1, layout.gen_tp * layout.gen_pp)
        bcast_world = 1 + layout.gen_world

        if bcast_world != 1 + tp * (layout.gen_world // tp):
            # gen_world should already be tp * (number of TP groups), but
            # surface the inconsistency loudly rather than silently
            # mis-rank the workers.
            logger.warning(
                "gen_world=%d, gen_tp=%d -- world is not a clean multiple of tp; "
                "rank assignment may not match AReaL convention",
                layout.gen_world,
                layout.gen_tp,
            )

        logger.info(
            "areal_xccl PG rendezvous: src=trainer_rank0 host=%s port=%d "
            "world=1+%d backend=%s group=%s",
            master_addr,
            master_port,
            layout.gen_world,
            self._bcast_backend,
            self._group_name,
        )

        # ---- Validate trainer endpoint surface --------------------------
        if not hasattr(trainer_actor, "init_xccl_source"):
            raise NotImplementedError(
                "ArealXcclBackend.initialize: trainer_actor is missing the "
                "'init_xccl_source' endpoint. See the contract in "
                "forge.engines.weight_sync.backends.areal_xccl module "
                "docstring; until that endpoint lands, use "
                "backend='collective_broadcast' (HiXL) or "
                "backend='torchstore_multi_vol'."
            )

        # ---- Fire trainer + worker init concurrently -------------------
        workers = await self._resolve_workers(generator_actor)

        async def _await(fut):
            return await fut

        async def _init_workers() -> Any:
            """Fanout init_bcast_group across vLLM TP workers.

            Worker ranks: ``1 + local_rank``.  Mirrors the rank scheme used
            by ``CollectiveBroadcastBackend`` (which puts storage at rank 0
            and workers at 1..TP) so both backends keep the same mental
            model.  For multi-server gen worlds (PP=1, multiple TP groups
            stacked into one mesh), the rank scheme extends naturally to
            ``rank = 1 + server_idx * tp + local_rank``; here we drive it
            off Monarch mesh rank.
            """
            worker_extent = workers.extent
            labels = list(worker_extent)

            # Identify the dim we iterate over to assign sequential ranks.
            # vLLM TP workers historically sit on a {hosts: 1, procs: tp}
            # mesh; for multi-host gen later we'll fold both dims into the
            # rank.  Single iter axis for now; raise on anything else so
            # the failure mode is loud and reviewable.
            iter_dim: str | None = None
            iter_size = 0
            fixed_dims: dict[str, int] = {}
            for lbl in labels:
                size = worker_extent[lbl]
                if size == layout.gen_world and iter_dim is None:
                    iter_dim = lbl
                    iter_size = size
                elif size == 1:
                    fixed_dims[lbl] = 0
                else:
                    raise AssertionError(
                        f"worker extent {worker_extent} has dim '{lbl}' of "
                        f"size {size}, neither 1 nor gen_world="
                        f"{layout.gen_world}; ArealXcclBackend doesn't "
                        "address this mesh shape yet"
                    )
            if iter_dim is None:
                raise AssertionError(
                    f"worker extent {worker_extent} has no dim of size "
                    f"gen_world={layout.gen_world}"
                )

            tasks = []
            for local_rank in range(iter_size):
                slice_kwargs = dict(fixed_dims)
                slice_kwargs[iter_dim] = local_rank
                w = workers.slice(**slice_kwargs)
                tasks.append(
                    _await(
                        w.init_bcast_group.call_one(
                            master_addr=master_addr,
                            master_port=master_port,
                            world_size=bcast_world,
                            rank=1 + local_rank,
                            backend=self._bcast_backend,
                            group_name=self._group_name,
                            timeout_s=self._bcast_timeout_s,
                        )
                    )
                )
            return await asyncio.gather(*tasks)

        try:
            trainer_init, worker_init = await asyncio.wait_for(
                asyncio.gather(
                    _await(
                        trainer_actor.init_xccl_source.call_one(
                            master_addr=master_addr,
                            master_port=master_port,
                            world_size=bcast_world,
                            rank=0,
                            backend=self._bcast_backend,
                            group_name=self._group_name,
                            timeout_s=self._bcast_timeout_s,
                        )
                    ),
                    _init_workers(),
                ),
                timeout=self._bcast_timeout_s + 30,
            )
        except TimeoutError as e:
            raise RuntimeError(
                f"ArealXcclBackend: HCCL group init timed out after "
                f"{self._bcast_timeout_s}s (addr={master_addr} "
                f"port={master_port} world={bcast_world}). Check that the "
                f"port is reachable cross-host and HCCL_CONNECT_TIMEOUT "
                f"allows enough time for RoCE link discovery."
            ) from e

        self._resolved_master_addr = master_addr
        self._resolved_master_port = master_port
        self._initialized = True

        logger.info(
            "ArealXcclBackend initialized: trainer=%s workers=%s",
            trainer_init,
            worker_init,
        )
        return {
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": bcast_world,
            "backend": self._bcast_backend,
            "group_name": self._group_name,
        }

    async def push(self, trainer_actor: Any, version: int) -> PushResult:
        """Run trainer-side gather + rank-0 broadcast on the xccl group.

        On AReaL's xccl path the broadcast IS the data transfer -- the
        moment trainer rank 0 emits ``dist.broadcast(flat, src=0, group)``,
        every joined worker's ``dist.broadcast`` recv into its own buffer
        is satisfied.  We launch the trainer's ``publish_weights_xccl``
        endpoint (which does the gather + broadcast) concurrently with
        the workers' ``recv_and_load_flat`` so the two halves of the
        collective land in the same wall-clock window.

        :meth:`pull` is then a metadata-only confirmation step that just
        returns the cached :class:`PullResult`.
        """
        if not self._initialized:
            raise RuntimeError("ArealXcclBackend.push: not initialized")

        if not hasattr(trainer_actor, "publish_weights_xccl"):
            raise NotImplementedError(
                "ArealXcclBackend.push: trainer_actor is missing the "
                "'publish_weights_xccl' endpoint. See module docstring for "
                "the expected signature."
            )

        workers = await self._resolve_workers(self._generator_actor)

        async def _await(fut):
            return await fut

        t0 = time.perf_counter()
        try:
            # We do NOT yet know plan/total_bytes -- the trainer endpoint
            # computes them as part of state_dict gather.  Workers join
            # the broadcast collective with a placeholder plan; the
            # trainer endpoint returns the resolved plan which we surface
            # in PushResult so :meth:`pull` can echo it.
            #
            # Implementation note: ``recv_and_load_flat`` requires
            # ``plan`` and ``total_bytes`` up-front (they size the recv
            # buffer).  AReaL's wire path threads these through HTTP
            # ``set_weight_meta`` BEFORE ``update_weight_xccl``.  Mirror
            # that here: do a pre-flight metadata exchange.
            plan, total_bytes, build_s = await self._fetch_plan(trainer_actor, version)

            broadcast_meta_mesh, worker_recv_mesh = await asyncio.gather(
                _await(
                    trainer_actor.publish_weights_xccl.call(
                        version=version,
                        group_name=self._group_name,
                    )
                ),
                _await(
                    workers.recv_and_load_flat.call(
                        version=version,
                        plan=plan,
                        total_bytes=total_bytes,
                        src_rank=0,
                    )
                ),
            )
        except Exception as e:
            logger.exception("push(v%d): xccl broadcast failed", version)
            raise RuntimeError(
                f"ArealXcclBackend.push({version}) failed: {type(e).__name__}: {e}"
            ) from e
        push_s = time.perf_counter() - t0

        per_rank: dict[int, dict] = {}
        broadcast_s_max = 0.0
        for _ref, payload in broadcast_meta_mesh.items():
            if not isinstance(payload, dict):
                continue
            rank = payload.get("rank", 0)
            per_rank[rank] = payload
            broadcast_s_max = max(broadcast_s_max, payload.get("broadcast_s", 0.0))

        per_worker: dict[int, dict] = {}
        workers_load_s = 0.0
        success_bytes = 0
        for _ref, payload in worker_recv_mesh.items():
            if not isinstance(payload, dict):
                continue
            per_worker[len(per_worker)] = payload
            workers_load_s = max(workers_load_s, payload.get("workers_load_s", 0.0))
            if payload.get("success"):
                success_bytes = max(success_bytes, payload.get("bytes", 0))

        any_fail = any(not p.get("success", True) for p in per_worker.values())
        num_keys = 0 if any_fail else len(plan)
        bytes_total = success_bytes or total_bytes

        # Cache for pull -- on this backend pull is a metadata echo, not
        # a separate transfer.
        self._cached_pull_result = PullResult(
            version=version,
            num_keys=num_keys,
            bytes_total=bytes_total,
            pull_s=0.0,
            workers_load_s=workers_load_s,
            per_worker=per_worker,
        )

        logger.info(
            "push(v%d): broadcast_s_max=%.2f push_s=%.2f workers_load_s=%.2f "
            "num_keys=%d bytes=%d",
            version,
            broadcast_s_max,
            push_s,
            workers_load_s,
            num_keys,
            bytes_total,
        )
        return PushResult(
            version=version,
            num_keys=num_keys,
            bytes_total=bytes_total,
            push_s=push_s,
            build_state_dict_s=build_s,
            per_rank=per_rank,
        )

    async def pull(self, generator_actor: Any, version: int) -> PullResult:
        """No-op: workers received inside :meth:`push`'s broadcast.

        We just echo the cached :class:`PullResult` so the
        :class:`WeightSyncService` accounting stays consistent with the
        other backends (which split push and pull into separate transfer
        phases).  ``pull_s`` is reported as 0 because the bytes are
        already on the workers by the time :meth:`push` returns.
        """
        if not self._initialized:
            raise RuntimeError("ArealXcclBackend.pull: not initialized")
        if (
            self._cached_pull_result is None
            or self._cached_pull_result.version != version
        ):
            raise RuntimeError(
                f"ArealXcclBackend.pull({version}): no cached PullResult; "
                f"push({version}) must run first on this backend"
            )
        return self._cached_pull_result

    async def shutdown(self) -> None:
        """Tear down the xccl ProcessGroup on both legs."""
        if not self._initialized:
            return

        # Generator workers: existing endpoint reused from collective_broadcast.
        try:
            workers = await self._resolve_workers(self._generator_actor)
            await workers.shutdown_bcast_group.call()
        except Exception:
            logger.exception("shutdown: worker-side shutdown_bcast_group failed")

        # Trainer side: optional endpoint; tolerate absence so shutdown
        # never raises in finally blocks.
        if self._trainer_actor is not None and hasattr(
            self._trainer_actor, "shutdown_xccl_source"
        ):
            try:
                await self._trainer_actor.shutdown_xccl_source.call_one(
                    group_name=self._group_name,
                )
            except Exception:
                logger.exception("shutdown: trainer-side shutdown_xccl_source failed")

        self._initialized = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _resolve_workers(self, generator_actor: Any) -> Any:
        """Return the vLLM worker ActorMesh reference.

        Mirrors :meth:`CollectiveBroadcastBackend._resolve_workers` so the
        two backends agree on how the generator exposes its workers.
        """
        if hasattr(generator_actor, "workers") and generator_actor.workers is not None:
            return generator_actor.workers
        getter = getattr(generator_actor, "get_worker_mesh", None)
        if getter is not None:
            return await getter.call_one()
        raise RuntimeError(
            "generator_actor does not expose a workers mesh handle; "
            "ArealXcclBackend needs one to fanout init_bcast_group / "
            "recv_and_load_flat"
        )

    async def _resolve_trainer_host(self, trainer_actor: Any) -> str:
        """Pick the hostname workers should dial for the rendezvous.

        Preference order:

        1. ``trainer_actor.get_host.call_one()`` if the trainer exposes
           a host-info endpoint.
        2. ``socket.gethostname()`` of the driver process -- works when
           the driver is colocated with trainer rank 0.
        """
        getter = getattr(trainer_actor, "get_host", None)
        if getter is not None:
            try:
                host = await getter.call_one()
                if isinstance(host, str) and host:
                    return host
            except Exception:
                logger.warning("trainer.get_host() failed; falling back to gethostname")
        return socket.gethostname()

    async def _fetch_plan(
        self,
        trainer_actor: Any,
        version: int,
    ) -> tuple[list, int, float]:
        """Pre-flight metadata pass: get the flat layout from the trainer.

        Mirrors AReaL's HTTP ``set_weight_meta`` step.  The trainer's
        ``publish_weights_xccl`` endpoint may also accept a
        ``return_plan_only=True`` mode that gathers the state_dict shape
        and returns the plan WITHOUT actually broadcasting.  Falls back
        to a separate ``plan_xccl_layout`` endpoint if available.

        Returns ``(plan, total_bytes, build_state_dict_s)`` so the worker
        ``recv_and_load_flat`` calls can size their recv buffers
        correctly before the broadcast collective fires.
        """
        plan_endpoint = getattr(trainer_actor, "plan_xccl_layout", None)
        if plan_endpoint is None:
            raise NotImplementedError(
                "ArealXcclBackend._fetch_plan: trainer_actor is missing "
                "'plan_xccl_layout'. The endpoint should return "
                "{'plan': [...], 'total_bytes': N, 'build_state_dict_s': S} "
                "computed by ``forge.engines.weight_sync._flat_layout."
                "build_meta_from_state_dict + plan_layout``. See module "
                "docstring for the expected signature."
            )
        result = await plan_endpoint.call_one(version=version)
        plan = result.get("plan", [])
        total_bytes = int(result.get("total_bytes", 0))
        build_s = float(result.get("build_state_dict_s", 0.0))
        return plan, total_bytes, build_s


# -------------------------------------------------------------------- helpers


def _find_free_port() -> int:
    """Pick a free TCP port for the HCCL TCPStore rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


__all__ = ["ArealXcclBackend"]
