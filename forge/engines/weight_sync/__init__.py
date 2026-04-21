"""Weight synchronization strategy implementations.

Available strategies:
    - ``NCCLWeightSync``: Collective broadcast over NCCL/HCCL process group.
    - ``CheckpointWeightSync``: Save to disk, reload on inference side.
    - ``HIXLWeightSync`` (stub): One-sided RDMA via Monarch HIXL library.
    - ``TorchstoreWeightSync``: torchstore + Monarch RDMA (HiXL on NPU).

Use ``create_weight_sync()`` to instantiate by method name.
"""

from __future__ import annotations

from typing import Any

from forge.core.weight_sync import WeightSyncConfig as WeightSyncConfig
from forge.core.weight_sync import WeightSyncMethod


def create_weight_sync(
    method: str | WeightSyncMethod,
    **kwargs: Any,
):
    """Factory for weight sync strategies.

    Args:
        method: Sync method name or enum (``"nccl"``, ``"checkpoint"``, ``"hixl"``).
        **kwargs: Strategy-specific options.

    Returns:
        An object implementing ``WeightSyncStrategy``.
    """
    if isinstance(method, str):
        method = WeightSyncMethod(method)

    if method == WeightSyncMethod.NCCL:
        from forge.engines.weight_sync.nccl_sync import NCCLWeightSync

        return NCCLWeightSync(**kwargs)
    if method == WeightSyncMethod.CHECKPOINT:
        from forge.engines.weight_sync.checkpoint_sync import CheckpointWeightSync

        return CheckpointWeightSync(**kwargs)
    if method == WeightSyncMethod.HIXL:
        from forge.engines.weight_sync.hixl_sync import HIXLWeightSync

        return HIXLWeightSync(**kwargs)
    if method == WeightSyncMethod.TORCHSTORE:
        from forge.engines.weight_sync.torchstore_sync import TorchstoreWeightSync

        return TorchstoreWeightSync(**kwargs)
    raise ValueError(f"Unknown weight sync method: {method!r}")
