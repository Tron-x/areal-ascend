"""Per-proc environment setter for torchstore storage volume procs.

Replaces the ``spawn_procs(bootstrap=...)`` pattern that
``_storage_bootstrap_factory`` used to rely on.  On our Monarch build the
bootstrap callback is silently skipped (see
``forge/docs/weight_sync.md`` §7.1 Bug B): we verified by writing marker
files from inside the bootstrap closure and confirmed they never appear,
which is the same symptom torchforge worked around by switching GPU path
to a post-spawn ``EnvSetter`` actor.

Usage (caller = ``grpo.py::_spawn_storage_mesh``)::

    storage_mesh = storage_hosts.spawn_procs(
        per_host={"procs": N}, name="torchstore_storage_multi_vol",
    )
    setter = storage_mesh.spawn("_storage_env_setter", StorageEnvSetter)
    await setter.setup.call(npu_base=npu_base, pool_mb=pool_mb)

Each vol proc's ``setup`` runs independently, reads its own
``LOCAL_RANK`` (set by Monarch's ``per_host={"procs": N}`` spawn path),
and sets ``ASCEND_RT_VISIBLE_DEVICES`` + related HiXL/HCCL/torchstore
env vars for that proc before ``torch_npu`` gets imported for real.

Why this works where the bootstrap closure didn't: the setter actor
call runs *after* ``spawn_procs`` returns and *before* the
``ts.initialize(mesh=...)`` call that triggers StorageVolume
registration + eventual HixlManagerActor init.  The lifecycle order
lets us write env into each proc before any torch_npu device
initialization pegs that proc to NPU 0.
"""

from __future__ import annotations

import os

from monarch.actor import Actor, current_rank, endpoint


class StorageEnvSetter(Actor):
    """Per-proc env setter for storage volume procs.

    Mirrors what the legacy ``_storage_bootstrap_factory`` closure *would*
    have done if ``spawn_procs(bootstrap=...)`` actually fired.  Kept as
    a dedicated tiny actor (rather than folded into MultiVolTorchstoreBackend)
    so the lifecycle is visible / debuggable independently.
    """

    @endpoint
    def setup(
        self,
        npu_base: int,
        pool_mb: int = 8192,
        storage_device: str = "npu:0",
    ) -> dict:
        """Apply the per-proc env vars for this storage vol.

        Args:
            npu_base: first physical NPU id on this host to assign to
                storage vols.  vol rank ``i`` lands on ``npu_base + i``.
            pool_mb: MonarchRDMA staging pool size in MiB (per-proc).
            storage_device: ``torch.device`` string the pool sits on
                from the proc's masked view.  ``"npu:0"`` is correct
                under ``ASCEND_RT_VISIBLE_DEVICES`` masking because the
                proc then only sees a single NPU, which is logically
                device 0 inside the proc.

        Returns:
            A small status dict so the caller can double-check that
            setup actually fired on each rank (pushing structured
            evidence up to the driver; see `forge/docs/weight_sync.md`
            §7.1 Bug B for why we don't trust bootstrap anymore).
        """
        local_rank = current_rank().rank
        dev_id = npu_base + local_rank

        # Masking (must happen before torch_npu is imported — torch_npu
        # reads this env exactly once at import time and caches the
        # visible device list).
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)

        # After masking, the proc's single visible NPU is logical
        # device 0; both MONARCH_NPU_DEVICE (consumed by
        # monarch_rdma::backend::hixl::manager_actor::resolve_device_id)
        # and torch.npu.set_device agree on that.
        os.environ["MONARCH_NPU_DEVICE"] = "0"

        # HiXL transport selection + HCCL port range (matches the old
        # bootstrap exactly).
        os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")

        # torchstore pool/staging knobs.  The module constants
        # ``_STORAGE_DEVICE`` and ``_POOL_BYTES`` are cached at
        # torchstore's own import time, so setting env here works only
        # because torchstore hasn't been imported into the proc yet
        # (this setter is the first thing the proc runs post-spawn).
        os.environ["TORCHSTORE_MONARCH_RDMA_EAGER_D2H"] = "0"
        os.environ["TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE"] = storage_device
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_POOL_MB", str(pool_mb))

        # Distributed / torch expectations that some downstream code
        # still reads as sanity-check.
        os.environ.setdefault("RANK", str(local_rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))

        # Now import torch + pin the device.  This is done lazily so
        # env-set ordering above is guaranteed.
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

        # Write a marker file so the driver can verify setup actually
        # ran on this proc (same evidence mechanism we used to prove
        # bootstrap wasn't running).
        try:
            with open(f"/tmp/forge_storage_env_rank{local_rank}.log", "w") as fp:
                fp.write(
                    f"local_rank={local_rank} dev_id={dev_id} "
                    f"device_count={torch.npu.device_count()} "
                    f"current_device={torch.npu.current_device()} "
                    f"visible_env={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}\n"
                )
        except Exception:
            pass

        return {
            "local_rank": local_rank,
            "physical_npu": dev_id,
            "masked_device_count": torch.npu.device_count(),
            "masked_current_device": torch.npu.current_device(),
        }
