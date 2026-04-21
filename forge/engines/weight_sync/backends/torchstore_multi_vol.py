"""Multi-volume torchstore backend for the WeightSyncService.

Topology (matches torchforge ``apps/grpo/main.py`` lines 126-139)::

    trainer_host (N NPUs)                 generator_host
    +-------------------+                 +---------------+
    | rank 0  NPU 0    -+--> vol 0 (NPU) -+--> gen workers|
    | rank 1  NPU 1    -+--> vol 1 (NPU) -+-->  ts.get    |
    | rank 2  NPU 2    -+--> vol 2 (NPU) -+--> (inplace)  |
    | rank 3  NPU 3    -+--> vol 3 (NPU) -+--> ...        |
    +-------------------+                 +---------------+
                   HiXL RoCE cross-node (N NICs in parallel)

N = ``layout.train_world``.  Storage volumes are spawned on the trainer host
mesh so the **put** path is intra-host (HCCS / PCIe) -- very fast -- and the
**pull** path goes cross-node with N RoCE NICs feeding the vLLM workers in
parallel.  Each volume holds one HiXL engine on its own NPU; torchstore's
``LocalRankStrategy`` routes trainer rank i's ``ts.put`` to volume i via
``LOCAL_RANK``.

Phase-1 pull: every vLLM worker pulls every key (TP=1), calling
``ts.get(key, inplace_tensor=self.model.get_parameter(name).data)`` inside
the worker proc so HiXL RDMA writes straight into the NPU-resident model
parameter with no CPU detour.  Parameters that vLLM's ``model.load_weights``
fuses (e.g. Qwen3 ``gate_up_proj`` is ``gate_proj || up_proj``) don't have a
matching ``named_parameters`` entry, so those fall back to the classic
``load_weights`` path.  See ``WorkerWrapper.pull_weights`` for the
per-param routing.

Phase-2 / 3 (not implemented here): TP-aware ``tensor_slice_spec`` so each
worker only pulls its shard.  Shard-at-put on the trainer (instead of
storing the full gathered tensor) is the bigger win for large models but
needs inverse ``sd_adapter.from_hf`` to avoid quadratic rename cost.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from forge.engines.weight_sync.service import (
    ParallelLayout,
    ParamMeta,
    PullResult,
    PushResult,
)

logger = logging.getLogger("MultiVolTorchstoreBackend")


def _storage_bootstrap_factory(npu_base: int):
    """Return a picklable bootstrap closure that pins a storage proc to an NPU.

    Monarch calls the closure inside every spawned proc **before** any
    ``import torch`` happens, so ``ASCEND_RT_VISIBLE_DEVICES`` takes effect
    cleanly.  We rely on ``LOCAL_RANK`` being set per-proc by Monarch's
    ``per_host={"procs": N}`` spawn path (each proc gets
    ``LOCAL_RANK=i`` for ``i in range(N)``) to assign distinct physical NPUs
    across storage ranks; if a given Monarch build doesn't populate
    ``LOCAL_RANK`` we fall back to 0 (single-NPU contention visible as a
    ``HcclCommPrepare`` port conflict the caller can diagnose from the logs).
    """

    def _bootstrap():
        import os as _os

        local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
        dev_id = npu_base + local_rank
        _os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        # After ASCEND_RT_VISIBLE_DEVICES masking, the only visible NPU is
        # always local index 0.
        _os.environ["MONARCH_NPU_DEVICE"] = "0"
        _os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        _os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        _os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        _os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        _os.environ["TORCHSTORE_MONARCH_RDMA_EAGER_D2H"] = "0"
        _os.environ["TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE"] = "npu:0"
        _os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_POOL_MB", "8192")
        _os.environ.setdefault("RANK", str(local_rank))
        _os.environ.setdefault("LOCAL_RANK", str(local_rank))
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


class MultiVolTorchstoreBackend:
    """N-volume torchstore backend, volumes colocated with trainer ranks.

    ``storage_npu_base`` picks where the N volumes land on the trainer host:
    ``0`` means NPU 0..N-1 (same cards as the trainer itself if trainer is
    ``per_host={"npus": N}`` starting at 0, causing two HiXL engines per NPU);
    ``train_world`` means NPU N..2N-1 (separate cards from the trainer, which
    is what we want for the 2x4-NPU-used case on an 8-NPU host).  Default is
    ``train_world`` (no collision).
    """

    def __init__(
        self,
        *,
        storage_mesh: Any | None = None,
        storage_npu_base: int | None = None,
        pool_mb: int = 8192,
        flat: bool = True,
    ) -> None:
        """
        Args:
            storage_mesh: an already-spawned Monarch ``ProcMesh`` that the
                backend should use as the torchstore volume host.  When
                provided, :meth:`initialize` skips its internal spawn and
                runs ``ts.initialize`` against this mesh directly.  This
                is the "injected storage" path -- the caller
                (``grpo.py`` today, or a YAML-driven launcher helper
                tomorrow) decides where storage lives, which host it sits
                on, and which NPUs it occupies.  The backend no longer
                has to know.  Leave ``None`` to keep the legacy
                "spawn storage on trainer host, NPU
                ``storage_npu_base..storage_npu_base+train_world-1``"
                behaviour (used while callers are still migrating).
            storage_npu_base: (legacy path only) first NPU id to pin
                volumes to on the trainer host.  ``None`` = auto
                (``train_world``).  Ignored when ``storage_mesh`` is
                provided -- the caller's spawn bootstrap already pinned
                NPUs by then.
            pool_mb: MonarchRDMA staging pool size per storage volume.
            flat: when True (default), use the single-flat-key fast path
                (``TrainerActor.publish_weights_flat`` +
                ``WorkerWrapper.pull_weights_flat``) -- one RDMA transfer
                of the packed state_dict, ~17 GB/s on RoCE.  When False,
                fall back to the per-parameter path
                (``publish_weights`` + ``pull_weights``) which is easier
                to debug but pays 311 * per-key overhead.
        """
        self._storage_mesh_injected = storage_mesh is not None
        self._storage_mesh = storage_mesh
        self._storage_npu_base = storage_npu_base
        self._pool_mb = pool_mb
        self._flat = flat
        self._layout: ParallelLayout | None = None
        self._initialized = False
        # Cache the ParamMeta published by the last push so the pull side
        # doesn't need another round trip to learn the parameter inventory.
        self._cached_meta_version: int | None = None
        self._cached_meta: list[ParamMeta] | None = None
        # For the flat path we also cache the plan (param name ->
        # offset/shape/dtype) so the pull side can unpack without
        # asking the trainer again.
        self._cached_plan: list | None = None
        self._cached_total_bytes: int = 0

    # ------------------------------------------------------------------ key

    def key_for(self, version: int, name: str) -> str:
        """Ping-pong storage key.

        Matches upstream torchforge's
        ``f"policy_ver_{version % 2:010d}.{name}"`` so a torchforge producer
        and our consumer (or vice-versa) could interoperate on the same
        torchstore if the layout permits.
        """
        return f"policy_ver_{version % 2:010d}.{name}"

    # --------------------------------------------------------------- init

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        layout: ParallelLayout,
        config: dict,
    ) -> dict:
        import torchstore as ts
        from torchstore.strategy import LocalRankStrategy
        from torchstore.transport import TransportType

        self._layout = layout
        num_vols = layout.train_world

        if self._storage_mesh_injected:
            # Caller already provisioned the storage mesh -- backend is a
            # pure data-plane client here.  The bootstrap that set up the
            # NPU pinning, HiXL env vars, and staging pool size is the
            # caller's responsibility (see
            # ``_storage_bootstrap_factory`` which the legacy path uses,
            # and which external spawners are expected to reuse).
            logger.info("using injected storage mesh (%d volumes assumed)", num_vols)
        else:
            # Legacy behaviour retained so callers that haven't migrated
            # to external spawning keep working unchanged.  Storage goes
            # on the trainer host mesh at NPU
            # ``storage_npu_base..storage_npu_base+num_vols-1``.
            from forge.provisioner import _get_provisioner

            if self._storage_npu_base is None:
                # Avoid NPU collision with the trainer: trainer uses
                # 0..N-1, we put storage on N..2N-1.
                self._storage_npu_base = num_vols

            provisioner = await _get_provisioner()
            trainer_hosts = await provisioner.get_host_mesh(layout.trainer_mesh_name)

            self._storage_mesh = trainer_hosts.spawn_procs(
                per_host={"procs": num_vols},
                name="torchstore_storage_multi_vol",
                bootstrap=_storage_bootstrap_factory(self._storage_npu_base),
            )
            logger.info(
                "spawned %d torchstore volumes on mesh %r (NPU range %d..%d)",
                num_vols,
                layout.trainer_mesh_name,
                self._storage_npu_base,
                self._storage_npu_base + num_vols - 1,
            )

        # LocalRankStrategy needs LOCAL_RANK set in the driver proc too so
        # it can pick a routing volume for any ts.get / ts.put issued here.
        import os as _os

        _os.environ.setdefault("LOCAL_RANK", "0")
        _os.environ.setdefault("RANK", "0")

        await ts.initialize(
            num_storage_volumes=num_vols,
            strategy=LocalRankStrategy(
                default_transport_type=TransportType.MonarchRDMA
            ),
            mesh=self._storage_mesh,
        )
        self._initialized = True
        logger.info(
            "MultiVolTorchstoreBackend initialized (num_vols=%d, pool=%d MB)",
            num_vols,
            self._pool_mb,
        )
        return {
            "num_volumes": num_vols,
            "storage_npu_base": self._storage_npu_base,
            "pool_mb": self._pool_mb,
        }

    # --------------------------------------------------------------- push

    async def push(self, trainer_actor: Any, version: int) -> PushResult:
        """Drive the trainer-side write.

        Two code paths, controlled by ``self._flat``:

        * **flat (default)**: single-key single-transfer fast path.
          Every trainer rank runs ``publish_weights_flat(version, key)``
          where ``key = f"policy_ver_{v%2:010d}.flat"``.  Rank 0 packs
          the gathered state_dict into a 2 MiB-aligned NPU buffer and
          issues one ``ts.put`` covering the whole thing.  Other ranks
          participate in the FSDP collective and early-return.

        * **per-param**: legacy fallback where every trainer rank calls
          ``ts.put_batch`` keyed per-parameter.  Useful for debugging
          since each ``ts.get(key)`` on the consumer side can be
          inspected independently, but pays 311 * handshake overhead.
        """
        if not self._initialized:
            raise RuntimeError("MultiVolTorchstoreBackend: not initialized")

        if self._flat:
            return await self._push_flat(trainer_actor, version)
        return await self._push_per_param(trainer_actor, version)

    async def _push_flat(self, trainer_actor: Any, version: int) -> PushResult:
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
        for ref, payload in push_result_mesh.items():
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
        # Also populate ParamMeta for API compatibility.
        self._cached_meta = [
            ParamMeta(
                name=entry[0],
                shape=tuple(entry[1]),
                dtype=entry[2],
                nbytes=entry[4],
            )
            for entry in plan
        ]

        return PushResult(
            version=version,
            num_keys=rank0_payload.get("num_keys", len(plan)),
            bytes_total=total_bytes,
            push_s=push_s,
            build_state_dict_s=build_s_max,
            per_rank=per_rank,
        )

    async def _push_per_param(self, trainer_actor: Any, version: int) -> PushResult:
        t0 = time.perf_counter()
        push_result_mesh = await trainer_actor.publish_weights.call(
            version=version, key_prefix=f"policy_ver_{version % 2:010d}"
        )
        push_s = time.perf_counter() - t0

        per_rank: dict[int, dict] = {}
        rank0_payload: dict | None = None
        total_bytes = 0
        build_s_max = 0.0
        for ref, payload in push_result_mesh.items():
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

        self._cached_meta_version = version
        self._cached_plan = None
        self._cached_total_bytes = 0
        self._cached_meta = [
            ParamMeta(name=n, shape=tuple(s), dtype=d, nbytes=nb)
            for n, s, d, nb in zip(
                rank0_payload.get("param_names", []),
                rank0_payload.get("param_shapes", []),
                rank0_payload.get("param_dtypes", []),
                rank0_payload.get("param_nbytes", []),
            )
        ]

        num_keys = rank0_payload.get("num_keys", len(self._cached_meta))
        return PushResult(
            version=version,
            num_keys=num_keys,
            bytes_total=total_bytes,
            push_s=push_s,
            build_state_dict_s=build_s_max,
            per_rank=per_rank,
        )

    def flat_key_for(self, version: int) -> str:
        return f"policy_ver_{version % 2:010d}.flat"

    # --------------------------------------------------------------- pull

    async def pull(self, generator_actor: Any, version: int) -> PullResult:
        """Drive the generator-side read.

        Two code paths, picked to match the push path cached by
        :meth:`push`:

        * **flat**: one ``ts.get(flat_key)`` per worker into an aligned
          flat buffer, then a local unpack into model params.  Near
          theoretical HiXL bandwidth because the pool-staging detour is
          avoided (the flat buffer is itself 2 MiB aligned).
        * **per-param**: 311 concurrent ``ts.get`` calls going through
          the staging pool, ~1 GB/s on our hardware.
        """
        if not self._initialized:
            raise RuntimeError("MultiVolTorchstoreBackend: not initialized")

        meta = self._cached_meta
        if meta is None or self._cached_meta_version != version:
            raise RuntimeError(
                f"MultiVolTorchstoreBackend.pull({version}): no cached "
                f"param meta; push({version}) must be called first to "
                "populate it"
            )

        if self._flat and self._cached_plan is not None:
            return await self._pull_flat(generator_actor, version)
        return await self._pull_per_param(generator_actor, version)

    async def _pull_flat(self, generator_actor: Any, version: int) -> PullResult:
        key = self.flat_key_for(version)
        total_bytes = self._cached_total_bytes
        plan = self._cached_plan or []

        t0 = time.perf_counter()
        pull_mesh = await generator_actor.update_weights_sync.call(
            version,
            "torchstore",
            {
                "flat_key": key,
                "flat_plan": plan,
                "total_bytes": total_bytes,
            },
        )
        pull_s = time.perf_counter() - t0

        workers_load_s = 0.0
        success_bytes = 0
        per_worker: dict[int, dict] = {}
        for ref, payload in pull_mesh.items():
            if not isinstance(payload, dict):
                continue
            per_worker[len(per_worker)] = payload
            workers_load_s = max(workers_load_s, payload.get("workers_load_s", 0.0))
            success_bytes = max(success_bytes, payload.get("bytes", 0))

        return PullResult(
            version=version,
            num_keys=len(plan) if success_bytes > 0 else 0,
            bytes_total=success_bytes or total_bytes,
            pull_s=pull_s,
            workers_load_s=workers_load_s,
            per_worker=per_worker,
        )

    async def _pull_per_param(self, generator_actor: Any, version: int) -> PullResult:
        meta = self._cached_meta or []
        items = [{"name": m.name, "key": self.key_for(version, m.name)} for m in meta]
        total_bytes = sum(m.nbytes for m in meta)

        t0 = time.perf_counter()
        pull_mesh = await generator_actor.update_weights_sync.call(
            version, "torchstore", {"items": items, "total_bytes": total_bytes}
        )
        pull_s = time.perf_counter() - t0

        workers_load_s = 0.0
        success_bytes = 0
        per_worker: dict[int, dict] = {}
        for ref, payload in pull_mesh.items():
            if not isinstance(payload, dict):
                continue
            per_worker[len(per_worker)] = payload
            workers_load_s = max(workers_load_s, payload.get("workers_load_s", 0.0))
            success_bytes = max(success_bytes, payload.get("bytes", 0))

        return PullResult(
            version=version,
            num_keys=len(items) if success_bytes > 0 else 0,
            bytes_total=success_bytes or total_bytes,
            pull_s=pull_s,
            workers_load_s=workers_load_s,
            per_worker=per_worker,
        )

    # --------------------------------------------------------------- teardown

    async def shutdown(self) -> None:
        if not self._initialized:
            return
        try:
            import torchstore as ts

            await ts.shutdown()
        except Exception:
            logger.exception("torchstore shutdown raised; continuing")
        self._initialized = False
        logger.info("MultiVolTorchstoreBackend shutdown")
