"""Collective-broadcast weight-sync backend (storage-as-reflector).

Design rationale and positioning in the backend zoo are spelled out in
``forge/docs/weight_sync.md §7.4``. In one picture::

    trainer 0..N-1                  storage vol 0                inference TP 0..M-1
    +-------------+                 +-----------+                +--------------+
    |  HiXL put   |---- RoCE --->   | hold full |----- HCCL ---->|  bcast recv  |
    |  (ts.put)   |   (M NICs)      | flat      |   bcast src=0  |  (no ts.get) |
    +-------------+                 +-----------+                +--------------+
                                          /|\
                                           |
                 (vol 1..N-1 idle on this backend's fast path;
                  used only if the caller enabled shard publish,
                  which this backend currently rejects -- see below)

Comparison to :class:`MultiVolTorchstoreBackend`:

* **Push side is identical** -- both reuse the trainer's
  ``publish_weights_flat`` endpoint with ``shard_publish=0``, so the
  trainer's HiXL-put traffic pattern is unchanged and already-measured
  (single-key fast path, ~20 GB/s steady).
* **Pull side is different**. Instead of every inference TP worker
  issuing its own ``ts.get`` against a storage volume (which scales
  HiXL client channels as ``TP × num_vols`` and triggers
  ``hixl_connect 103901`` at TP>1), the storage volume acts as the
  **source of one HCCL broadcast** that fans out to every TP worker.
  Concurrent connection count = ``1 + TP``, handled by HCCL's QP
  aggregation -- unchanged regardless of TP value.

MVP limitations (documented here, fixable in follow-ups):

1. **No shard publish**. This backend requires the full flat tensor
   to live on a single storage volume (by default vol 0) so that vol
   can be the single bcast source.  When the caller wants shard
   publish, ``initialize`` raises.  Supporting multi-vol bcast (one
   HCCL group per storage vol, inference TP worker joins all N
   groups and concatenates N bcast buffers into the full flat) is a
   straightforward extension but not landed yet.
2. **TP group reuse across versions**. We build the bcast
   ``ProcessGroup`` once in ``initialize`` and reuse it for every
   pull. Weight-sync frequency (once per step) is low enough that
   re-init overhead doesn't matter, but we save a ~30s HCCL setup
   per sync cycle anyway.
3. **Direct-from-volume bcast**. The bcast source pulls the tensor
   out of ``InMemoryStore.kv[key]`` by reference -- no extra copy.
   This is safe because the storage vol's ``ts.put`` handler has
   already settled the tensor on the vol's NPU by the time ``pull``
   runs (push_and_pull is sequential in the service control loop).

The three MVP scripts that validated the transport story behind this
backend (``forge/scripts/test_cross_mesh_bcast*.py``) remain the
ground-truth reference for the HCCL shape and the coexistence
properties; they exercise the same endpoints this backend now wires
up into the production control plane.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from typing import Any

from forge.engines.weight_sync.service import (
    ParallelLayout,
    ParamMeta,
    PullResult,
    PushResult,
)

logger = logging.getLogger("CollectiveBroadcastBackend")


class CollectiveBroadcastBackend:
    """Push via HiXL (unchanged), pull via HCCL broadcast."""

    def __init__(
        self,
        *,
        storage_mesh: Any | None = None,
        storage_npu_base: int | None = None,
        pool_mb: int = 8192,
        bcast_src_vol_idx: int = 0,
        bcast_master_port: int | None = None,
        bcast_timeout_s: int = 180,
        bcast_backend: str = "hccl",
        store_name: str | None = None,
    ) -> None:
        """
        Args:
            storage_mesh: Monarch ``ProcMesh`` to spawn the storage
                volumes on. Same contract as ``MultiVolTorchstoreBackend``
                -- when provided, the backend skips its internal spawn
                and reuses the caller's mesh.  Leave ``None`` to fall
                back to the legacy self-spawning behavior on the
                trainer host.
            storage_npu_base: legacy self-spawn mode only; NPU offset
                for the storage volumes.
            pool_mb: MonarchRDMA staging pool MiB per volume.
            bcast_src_vol_idx: Which storage volume acts as the HCCL
                broadcast source. Default 0 (first volume). Must be in
                ``[0, train_world)``.
            bcast_master_port: TCP port on the storage host used for
                the HCCL TCPStore rendezvous. ``None`` picks a
                free port at ``initialize`` time.
            bcast_timeout_s: HCCL init / broadcast timeout.
            bcast_backend: ``torch.distributed`` backend for the bcast
                group. Default ``"hccl"`` on NPU; set ``"nccl"`` on
                CUDA builds.
            store_name: torchstore instance name (rarely overridden).
        """
        self._storage_mesh_injected = storage_mesh is not None
        self._storage_mesh = storage_mesh
        self._storage_npu_base = storage_npu_base
        self._pool_mb = pool_mb
        self._bcast_src_vol_idx = bcast_src_vol_idx
        self._bcast_master_port = bcast_master_port
        self._bcast_timeout_s = bcast_timeout_s
        self._bcast_backend = bcast_backend
        self._store_name = store_name

        self._layout: ParallelLayout | None = None
        self._initialized = False
        self._bcast_initialized = False

        self._storage_volumes: Any | None = None  # StorageVolume ActorMesh
        self._generator_actor: Any | None = None
        self._cached_meta_version: int | None = None
        self._cached_plan: list | None = None
        self._cached_total_bytes: int = 0
        self._cached_meta: list[ParamMeta] | None = None

    # --------------------------------------------------------------- key
    def flat_key_for(self, version: int) -> str:
        """Same ping-pong key shape as MultiVolTorchstoreBackend so the
        trainer side can use the same ``publish_weights_flat`` endpoint
        verbatim (it just publishes to vol 0 when shard publish is off).
        """
        return f"policy_ver_{version % 2:010d}.flat"

    # --------------------------------------------------------------- init

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        layout: ParallelLayout,
        config: dict,
    ) -> dict:
        """Set up torchstore + the HCCL broadcast group.

        Steps:

        1. Spawn / reuse storage ProcMesh (same as multi_vol).
        2. Manually spawn ``StorageVolume`` actors (so we keep a
           reference; ``ts.initialize`` also spawns, but throws away
           the mesh so we can't reach individual vols from here).
        3. Register our spawned vols into the torchstore Controller.
        4. Rendezvous the bcast vol + every vLLM TP worker into one
           ``torch.distributed`` group; rank 0 = storage vol, rank 1..TP
           = TP workers in Monarch mesh-rank order.
        """
        import os as _os

        import torchstore as ts
        from monarch.actor import get_or_spawn_controller
        from torchstore.api import DEFAULT_TORCHSTORE_NAME
        from torchstore.controller import Controller
        from torchstore.storage_volume import StorageVolume
        from torchstore.strategy import LocalRankStrategy
        from torchstore.transport import TransportType

        self._layout = layout
        self._generator_actor = generator_actor
        num_vols = layout.train_world
        tp = max(1, layout.gen_tp * layout.gen_pp)
        store_name = self._store_name or DEFAULT_TORCHSTORE_NAME

        # Sanity: shard_publish must be off for MVP (see class docstring).
        if int(_os.environ.get("FORGE_SHARD_PUBLISH", "0")) == 1:
            raise RuntimeError(
                "CollectiveBroadcastBackend (MVP) requires "
                "FORGE_SHARD_PUBLISH=0. Shard publish fragments the flat "
                "tensor across N storage vols, but the bcast source "
                "needs the full tensor in one place. Multi-vol bcast is "
                "tracked as a follow-up; unset the env var to use this "
                "backend today."
            )

        # ---- Storage mesh --------------------------------------------
        if self._storage_mesh_injected:
            logger.info("using injected storage mesh (%d volumes assumed)", num_vols)
        else:
            from forge.engines.weight_sync.backends.torchstore_multi_vol import (
                _storage_bootstrap_factory,
            )
            from forge.provisioner import _get_provisioner

            if self._storage_npu_base is None:
                self._storage_npu_base = num_vols
            provisioner = await _get_provisioner()
            trainer_hosts = await provisioner.get_host_mesh(layout.trainer_mesh_name)
            self._storage_mesh = trainer_hosts.spawn_procs(
                per_host={"procs": num_vols},
                name="torchstore_storage_multi_vol",
                bootstrap=_storage_bootstrap_factory(self._storage_npu_base),
            )
            logger.info(
                "spawned %d storage volumes at NPU %d..%d",
                num_vols,
                self._storage_npu_base,
                self._storage_npu_base + num_vols - 1,
            )

        # ---- torchstore init (manual, so we retain vol refs) ---------
        _os.environ.setdefault("LOCAL_RANK", "0")
        _os.environ.setdefault("RANK", "0")
        strategy = LocalRankStrategy(default_transport_type=TransportType.MonarchRDMA)
        self._storage_volumes = await StorageVolume.spawn(
            num_volumes=num_vols,
            mesh=self._storage_mesh,
            id_func=strategy.get_volume_id,
        )
        controller = await get_or_spawn_controller(store_name, Controller)
        await controller.init.call(
            strategy=strategy,
            num_storage_volumes=num_vols,
            storage_volumes=self._storage_volumes,
        )
        # ts.put / ts.get in any proc (trainer, driver, vol) will
        # lookup this controller by name via get_or_spawn_controller
        # and reuse it. No additional ``ts.initialize`` call is needed
        # -- calling it would re-invoke ``controller.init`` and raise
        # "TorchStore is already initialized".
        self._initialized = True
        # Silence unused-import: ts is used indirectly by downstream
        # callers that ``from torchstore import *`` via this module's
        # transitive imports.  Keeping the top-level import makes the
        # dependency graph explicit.
        _ = ts

        # ---- HCCL broadcast group rendezvous -------------------------
        # 1. Pick the bcast source vol.  Its host + a free TCP port
        #    become the rendezvous point.
        bcast_vol = self._storage_volumes.slice(
            **{k: self._bcast_src_vol_idx for k in list(self._storage_volumes.extent)}
        )

        # Ask the bcast vol which host it's on -- we need the hostname
        # for the HCCL TCPStore init_method.  ``StorageVolume.get_id``
        # (torchstore upstream) returns ``(volume_id, hostname)`` --
        # note the order, we want element [1] for the hostname.
        vol_info = await bcast_vol.get_id.call_one()
        if isinstance(vol_info, tuple) and len(vol_info) >= 2:
            master_addr = vol_info[1]
        elif isinstance(vol_info, tuple):
            master_addr = vol_info[0]
        else:
            master_addr = str(vol_info)
        # Sanity: a ``volume_id`` sneaking through would be an opaque
        # hex string like ``StorageVolumes_1f23ab`` -- not a hostname --
        # and would fail DNS.  Fall back to this_host's hostname if
        # the vol's reported hostname looks synthetic.
        if not master_addr or master_addr.startswith("StorageVolumes_"):
            master_addr = socket.gethostname()
        master_port = self._bcast_master_port or _find_free_port(master_addr)

        bcast_world = 1 + tp
        logger.info(
            "HCCL bcast PG rendezvous: src_vol=%d host=%s port=%d world=1+%d",
            self._bcast_src_vol_idx,
            master_addr,
            master_port,
            tp,
        )

        # 2. Fire init_bcast_group on both legs concurrently.
        #    Storage side: one endpoint, rank 0.
        #    Worker side: fanout across TP workers, rank 1..TP assigned
        #    from Monarch mesh rank.
        async def _await(fut):
            return await fut

        workers = await self._resolve_workers(generator_actor)

        async def _init_workers() -> dict:
            # vLLM workers live on a ``{hosts: 1, procs: tp}`` 2D mesh
            # (single host, tp ranks).  Identify the dim of size ``tp``
            # (the TP rank axis) and slice all remaining dims to 0.
            # If the layout ever grows to TP-across-hosts, this loop
            # surfaces it as a clear assertion rather than a silent
            # mis-addressing.
            worker_extent = workers.extent
            labels = list(worker_extent)
            procs_like: str | None = None
            fixed_dims: dict[str, int] = {}
            for lbl in labels:
                size = worker_extent[lbl]
                if size == tp and procs_like is None:
                    procs_like = lbl
                elif size == 1:
                    fixed_dims[lbl] = 0
                else:
                    raise AssertionError(
                        f"worker extent {worker_extent} has dim '{lbl}' "
                        f"of size {size}, neither 1 nor tp={tp}; "
                        "CollectiveBroadcastBackend doesn't address this"
                    )
            if procs_like is None:
                raise AssertionError(
                    f"worker extent {worker_extent} has no dim of size "
                    f"tp={tp}; cannot identify the TP dimension"
                )
            worker_tasks = []
            for local_rank in range(tp):
                slice_kwargs = dict(fixed_dims)
                slice_kwargs[procs_like] = local_rank
                w = workers.slice(**slice_kwargs)
                worker_tasks.append(
                    _await(
                        w.init_bcast_group.call_one(
                            master_addr=master_addr,
                            master_port=master_port,
                            world_size=bcast_world,
                            rank=1 + local_rank,
                            backend=self._bcast_backend,
                            timeout_s=self._bcast_timeout_s,
                        )
                    )
                )
            return await asyncio.gather(*worker_tasks)

        try:
            storage_init, worker_init = await asyncio.wait_for(
                asyncio.gather(
                    _await(
                        bcast_vol.init_bcast_group.call_one(
                            master_addr=master_addr,
                            master_port=master_port,
                            world_size=bcast_world,
                            rank=0,
                            backend=self._bcast_backend,
                            timeout_s=self._bcast_timeout_s,
                        )
                    ),
                    _init_workers(),
                ),
                timeout=self._bcast_timeout_s + 30,
            )
        except TimeoutError as e:
            raise RuntimeError(
                f"CollectiveBroadcastBackend: HCCL bcast group init timed "
                f"out after {self._bcast_timeout_s}s (addr={master_addr} "
                f"port={master_port} world={bcast_world}). Check that "
                f"port {master_port} is reachable cross-host and "
                f"HCCL_CONNECT_TIMEOUT allows enough time for the RoCE "
                f"link discovery."
            ) from e

        self._bcast_master_port = master_port  # freeze the resolved port
        self._bcast_initialized = True

        logger.info(
            "CollectiveBroadcastBackend initialized: storage=%s workers=%s",
            storage_init,
            worker_init,
        )
        return {
            "num_volumes": num_vols,
            "bcast_src_vol_idx": self._bcast_src_vol_idx,
            "bcast_master_addr": master_addr,
            "bcast_master_port": master_port,
            "bcast_world_size": bcast_world,
            "storage_npu_base": self._storage_npu_base,
            "pool_mb": self._pool_mb,
        }

    # --------------------------------------------------------------- push
    async def push(self, trainer_actor: Any, version: int) -> PushResult:
        """HiXL put to storage -- same endpoint as multi_vol.

        Since shard_publish is off (enforced in ``initialize``),
        ``publish_weights_flat`` collapses to the legacy rank-0-only
        path: rank 0 packs the full state_dict and ts.put's it under
        the single ``flat_key``, which lands in storage vol 0 (the
        bcast source). Other ranks participate in the FSDP gather and
        return metadata only.
        """
        if not self._initialized:
            raise RuntimeError("CollectiveBroadcastBackend: not initialized")

        key = self.flat_key_for(version)
        t0 = time.perf_counter()
        push_result_mesh = await trainer_actor.publish_weights_flat.call(
            version=version, key=key
        )
        push_s = time.perf_counter() - t0

        per_rank: dict[int, dict] = {}
        rank0_payload: dict | None = None
        total_bytes = 0
        build_s_max = 0.0
        for _ref, payload in push_result_mesh.items():
            if not isinstance(payload, dict):
                continue
            rank = payload.get("rank", 0)
            per_rank[rank] = payload
            if rank == 0:
                rank0_payload = payload
            total_bytes = max(total_bytes, payload.get("bytes", 0))
            build_s_max = max(build_s_max, payload.get("build_state_dict_s", 0.0))

        if rank0_payload is None:
            rank0_payload = next(iter(per_rank.values())) if per_rank else {}
        plan = rank0_payload.get("plan", [])

        self._cached_meta_version = version
        self._cached_plan = plan
        self._cached_total_bytes = total_bytes
        self._cached_meta = [
            ParamMeta(
                name=entry[0],
                shape=tuple(entry[1]),
                dtype=entry[2],
                nbytes=entry[4],
            )
            for entry in plan
        ]

        put_s_max = max((p.get("put_s", 0.0) for p in per_rank.values()), default=0.0)
        logger.info(
            "push(v%d): ranks_seen=%s plan_len=%d total_bytes=%d put_s_max=%.2f",
            version,
            sorted(per_rank.keys()),
            len(plan),
            total_bytes,
            put_s_max,
        )
        return PushResult(
            version=version,
            num_keys=rank0_payload.get("num_keys", len(plan)),
            bytes_total=total_bytes,
            push_s=push_s,
            build_state_dict_s=build_s_max,
            per_rank=per_rank,
        )

    # --------------------------------------------------------------- pull
    async def pull(self, generator_actor: Any, version: int) -> PullResult:
        """HCCL bcast from storage vol 0 -> inference TP workers.

        Concurrent with the storage-side bcast, all TP workers call
        ``recv_and_load_flat`` which allocates a 2 MiB-aligned flat
        buffer, receives the broadcast, unpacks into model parameters
        (direct copy_ or vLLM load_weights), and retains the flat
        buffer until next pull.
        """
        if not self._bcast_initialized:
            raise RuntimeError(
                "CollectiveBroadcastBackend.pull: bcast group not initialized; "
                "call initialize() first"
            )
        if self._cached_meta_version != version or self._cached_plan is None:
            raise RuntimeError(
                f"CollectiveBroadcastBackend.pull({version}): no cached "
                "plan; push() must be called first"
            )

        key = self.flat_key_for(version)
        total_bytes = self._cached_total_bytes
        plan = self._cached_plan or []

        bcast_vol = self._storage_volumes.slice(
            **{k: self._bcast_src_vol_idx for k in list(self._storage_volumes.extent)}
        )
        workers = await self._resolve_workers(generator_actor)

        async def _await(fut):
            return await fut

        t0 = time.perf_counter()
        try:
            storage_r, worker_r_mesh = await asyncio.gather(
                _await(bcast_vol.bcast_tensor.call_one(key=key)),
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
            logger.exception("pull(v%d): bcast failed", version)
            raise RuntimeError(
                f"CollectiveBroadcastBackend.pull({version}) failed: "
                f"{type(e).__name__}: {e}"
            ) from e
        pull_s = time.perf_counter() - t0

        per_worker: dict[int, dict] = {}
        workers_load_s = 0.0
        success_bytes = 0
        for ref, payload in worker_r_mesh.items():
            if not isinstance(payload, dict):
                continue
            per_worker[len(per_worker)] = payload
            workers_load_s = max(workers_load_s, payload.get("workers_load_s", 0.0))
            if payload.get("success"):
                success_bytes = max(success_bytes, payload.get("bytes", 0))

        any_fail = any(not p.get("success", True) for p in per_worker.values())
        num_keys = 0 if any_fail else (len(plan) if success_bytes > 0 else 0)

        logger.info(
            "pull(v%d): storage_bcast_s=%.2f pull_s=%.2f workers_load_s=%.2f "
            "num_keys=%d bytes=%d",
            version,
            storage_r.get("bcast_s", 0.0),
            pull_s,
            workers_load_s,
            num_keys,
            success_bytes or total_bytes,
        )
        return PullResult(
            version=version,
            num_keys=num_keys,
            bytes_total=success_bytes or total_bytes,
            pull_s=pull_s,
            workers_load_s=workers_load_s,
            per_worker=per_worker,
        )

    # --------------------------------------------------------------- util

    async def _resolve_workers(self, generator_actor: Any) -> Any:
        """Return the vLLM worker ActorMesh reference.

        Generator spawns its vllm_workers ActorMesh inside its executor
        and stashes a handle at ``generator.workers``.  Reading it via
        the actor endpoint avoids racing with executor lifecycle.
        """
        if hasattr(generator_actor, "workers") and generator_actor.workers is not None:
            return generator_actor.workers
        # Fallback: ask the generator for its worker handle.
        getter = getattr(generator_actor, "get_worker_mesh", None)
        if getter is not None:
            return await getter.call_one()
        raise RuntimeError(
            "generator_actor does not expose a workers mesh handle; "
            "CollectiveBroadcastBackend needs one to fanout "
            "init_bcast_group / recv_and_load_flat"
        )

    # --------------------------------------------------------------- shutdown
    async def shutdown(self) -> None:
        if not self._bcast_initialized:
            return
        try:
            bcast_vol = self._storage_volumes.slice(
                **{
                    k: self._bcast_src_vol_idx
                    for k in list(self._storage_volumes.extent)
                }
            )
            await bcast_vol.shutdown_bcast_group.call_one()
        except Exception:
            logger.exception("shutdown: storage-side shutdown_bcast_group failed")

        try:
            workers = await self._resolve_workers(self._generator_actor)
            await workers.shutdown_bcast_group.call()
        except Exception:
            logger.exception("shutdown: worker-side shutdown_bcast_group failed")

        try:
            import torchstore as ts

            await ts.shutdown(self._store_name or None)
        except Exception:
            logger.exception("shutdown: ts.shutdown failed")

        self._bcast_initialized = False
        self._initialized = False


# -------------------------------------------------------------------- helpers


def _find_free_port(host: str) -> int:
    """Pick a free TCP port on ``host`` for the HCCL TCPStore.

    We bind on INADDR_ANY on whichever proc runs this, pull the picked
    port, and trust that ``host`` is reachable from the peers. The
    actual server socket is opened by ``init_process_group`` inside the
    storage volume proc later -- this helper just reserves a number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])
