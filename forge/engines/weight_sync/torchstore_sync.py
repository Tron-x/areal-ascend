"""TorchstoreWeightSync -- legacy single-volume torchstore driver.

**Superseded by** :mod:`forge.engines.weight_sync.service` +
:mod:`forge.engines.weight_sync.backends.torchstore_multi_vol`.  That path
uses N storage volumes colocated with trainer ranks (``LocalRankStrategy``),
lets vLLM workers pull directly into NPU params via ``ts.get(inplace_tensor=
model_param.data)``, and exposes a pluggable backend interface so future
topologies (P2P, dedicated PS, AReaL xccl fallback) slot in without touching
the Trainer / Generator / driver code.

This module is kept only for:

1. **Key-scheme helpers** (``get_param_key``, ``get_param_prefix``,
   ``extract_param_name``) that the new code reuses verbatim so the wire
   format of stored weights stays stable across the old and new paths.

2. **``TorchstoreWeightSync`` class** as a fallback implementation of the
   ``WeightSyncStrategy`` protocol.  ``forge/apps/grpo.py`` no longer routes
   ``FORGE_WEIGHT_SYNC=torchstore`` here (that now dispatches to the new
   service), but out-of-tree callers that build a ``WeightSyncConfig`` and
   pass a ``storage_mesh`` in ``extra`` will still work.

New code should import from ``forge.engines.weight_sync.service`` and
``forge.engines.weight_sync.backends`` instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from forge.core.weight_sync import WeightSyncConfig

logger = logging.getLogger("TorchstoreWeightSync")

KEY_DELIM = "."
VERSION_A = 0
VERSION_B = 1


def get_storage_version(step: int) -> int:
    """Map a monotonic step to a ping-pong storage slot (0 or 1)."""
    return VERSION_A if step % 2 == 0 else VERSION_B


def get_param_prefix(policy_version: int) -> str:
    return f"policy_ver_{get_storage_version(policy_version):010d}"


def get_param_key(policy_version: int, name: str) -> str:
    """``policy_ver_0000000000.{name}`` - compatible with torchforge's scheme.

    Keeping the exact same key format makes future interop with any
    torchforge-produced storage trivial.
    """
    return f"{get_param_prefix(policy_version)}{KEY_DELIM}{name}"


def extract_param_name(key: str) -> str:
    return KEY_DELIM.join(key.split(KEY_DELIM)[1:])


class TorchstoreWeightSync:
    """WeightSyncStrategy using torchstore + Monarch RDMA (HiXL on NPU).

    Requires the caller (normally the Forge driver) to supply a storage
    ``ProcMesh`` via ``config.extra['storage_mesh']``; the strategy will
    call ``ts.initialize`` on it.  Trainer and generator actors must
    provide ``push_weights_torchstore(version)`` and
    ``pull_weights_torchstore(version)`` endpoints respectively.  These
    run inside the actor processes (which share the driver's torchstore
    controller) and do the real ``ts.put_batch`` / ``ts.get`` work.
    """

    def __init__(self) -> None:
        self._trainer: Any = None
        self._generator: Any = None
        self._config: WeightSyncConfig | None = None
        self._storage_mesh: Any = None
        self._initialized = False
        self._current_version = 0

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        config: WeightSyncConfig,
    ) -> dict:
        storage_mesh = config.extra.get("storage_mesh") if config.extra else None
        if storage_mesh is None:
            raise RuntimeError(
                "TorchstoreWeightSync requires config.extra['storage_mesh'] "
                "(a Monarch ProcMesh on which to host the storage volumes)."
            )
        num_storage_volumes = (
            config.extra.get("num_storage_volumes", 1) if config.extra else 1
        )

        # LocalRankStrategy reads LOCAL_RANK / RANK from env.  The driver
        # process that calls ts.initialize needs at least one of them set
        # so the strategy can compute a volume id; downstream actors have
        # these set by their NPU bootstrap.
        import os as _os

        import torchstore as ts
        from torchstore.strategy import LocalRankStrategy
        from torchstore.transport import TransportType

        _os.environ.setdefault("LOCAL_RANK", "0")
        _os.environ.setdefault("RANK", "0")

        # Single NPU storage topology: trainer/generator both speak MonarchRDMA
        # (HiXL RoCE on NPU) straight into/out of the storage volume's NPU
        # device memory.  No CPU staging, no TCP fallback — that's the whole
        # point of this phase, and also the only config where we can
        # meaningfully measure HiXL bandwidth.
        await ts.initialize(
            num_storage_volumes=num_storage_volumes,
            strategy=LocalRankStrategy(
                default_transport_type=TransportType.MonarchRDMA
            ),
            mesh=storage_mesh,
        )

        self._trainer = trainer_actor
        self._generator = generator_actor
        self._config = config
        self._storage_mesh = storage_mesh
        self._initialized = True
        logger.info(
            "TorchstoreWeightSync initialized (storage volume on %s)",
            getattr(storage_mesh, "host_name", "<storage_mesh>"),
        )
        return {"status": "initialized", "method": "torchstore"}

    async def push(self, version: int) -> dict:
        """Push weights for ``version``, then tell the generator to pull.

        Returns per-side timing; driver should log these for bandwidth
        accounting.  Both sides run in parallel where possible, but to
        keep semantics simple (and avoid racing against the next training
        step's writes) we push first then pull.
        """
        if not self._initialized:
            raise RuntimeError("TorchstoreWeightSync not initialized")

        push_t0 = time.perf_counter()
        push_result = await self._trainer.push_weights_torchstore.call(
            policy_version=version
        )
        push_time = time.perf_counter() - push_t0

        # ``push_result`` is a mesh dict {rank_ref: trainer_result}.  We
        # care about rank 0's result because only rank 0 actually pushes
        # to torchstore (see TrainerActor.push_weights_torchstore); other
        # ranks just participate in the FSDP gather collective and return
        # metadata.  Pick rank 0 when present, fall back to the first.
        pr = None
        for _k, v in push_result.items():
            if isinstance(v, dict) and v.get("rank", 0) == 0:
                pr = v
                break
        if pr is None:
            _, pr = next(iter(push_result.items()))
        num_keys = pr.get("num_keys", 0)
        push_bytes = pr.get("bytes", 0)
        push_build_s = pr.get("build_state_dict_s", 0.0)
        payload = {
            "param_names": pr.get("param_names", []),
            "param_shapes": pr.get("param_shapes", []),
            "param_dtypes": pr.get("param_dtypes", []),
        }

        pull_t0 = time.perf_counter()
        pull_result = await self._generator.update_weights_sync.call(
            version, "torchstore", payload
        )
        pull_time = time.perf_counter() - pull_t0

        _, pullr = next(iter(pull_result.items()))
        pull_bytes = pullr.get("bytes", push_bytes)
        ok = pullr.get("success", False)

        if ok:
            self._current_version = version

        total_bytes = max(push_bytes, pull_bytes)
        summary = {
            "version": version,
            "success": ok,
            "num_keys": num_keys,
            "bytes": total_bytes,
            "push_s": push_time,
            "pull_s": pull_time,
            "total_s": push_time + pull_time,
            "build_state_dict_s": push_build_s,
            "push_gbps": (
                total_bytes / push_time / (1024**3) if push_time > 0 else 0.0
            ),
            "pull_gbps": (
                total_bytes / pull_time / (1024**3) if pull_time > 0 else 0.0
            ),
        }
        logger.info(
            "torchstore sync v%d: %d keys, %.2f GB, "
            "push=%.2fs (%.2f GB/s) pull=%.2fs (%.2f GB/s)",
            version,
            num_keys,
            total_bytes / (1024**3),
            push_time,
            summary["push_gbps"],
            pull_time,
            summary["pull_gbps"],
        )
        return summary

    async def get_status(self) -> dict:
        return {
            "method": "torchstore",
            "initialized": self._initialized,
            "current_version": self._current_version,
        }

    async def shutdown(self) -> None:
        if not self._initialized:
            return
        import torchstore as ts

        try:
            await ts.shutdown()
        except Exception:
            logger.exception("torchstore shutdown raised; continuing")
        self._initialized = False
        logger.info("TorchstoreWeightSync shutdown")
