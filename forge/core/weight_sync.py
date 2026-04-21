"""Weight synchronization protocols and data types.

Defines the contract for transferring updated model weights from
a training engine to inference engines (vLLM Generator).

Three strategies are supported:

1. **NCCL/XCCL**: Collective broadcast over a shared process group.
   Best for co-located training and inference on the same cluster.

2. **Checkpoint**: Trainer saves to disk/shared-fs, Generator reloads.
   Best for decoupled deployments (cross-cluster, elastic scheduling).

3. **HIXL**: One-sided RDMA via Monarch's HIXL library.
   Best for high-performance NPU clusters with Monarch transport.

Usage::

    strategy = create_weight_sync("nccl", trainer=trainer, generator=generator)
    await strategy.initialize(config={...})
    # After each training step:
    await strategy.push(version=step + 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class WeightSyncMethod(Enum):
    """Supported weight synchronization methods."""

    NCCL = "nccl"
    CHECKPOINT = "checkpoint"
    HIXL = "hixl"
    TORCHSTORE = "torchstore"


@dataclass
class WeightsSpec:
    """Describes model parameters for weight sync negotiation.

    The training engine produces this so the sync strategy knows
    what to transfer without depending on framework internals.

    Attributes:
        param_names: Ordered list of parameter names.
        param_shapes: Shape of each parameter (matching param_names order).
        param_dtypes: String dtype of each parameter (e.g. "bfloat16").
        total_params: Total number of scalar parameters.
        model_path: Original model identifier (for checkpoint reload).
        extra: Backend-specific metadata (e.g. sharding info for FSDP).
    """

    param_names: list[str] = field(default_factory=list)
    param_shapes: list[tuple[int, ...]] = field(default_factory=list)
    param_dtypes: list[str] = field(default_factory=list)
    total_params: int = 0
    model_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightSyncConfig:
    """Configuration for a weight sync strategy.

    Attributes:
        method: Sync method to use.
        checkpoint_dir: Directory for checkpoint-based sync.
        group_name: NCCL process group name.
        master_addr: Address for NCCL rendezvous.
        master_port: Port for NCCL rendezvous.
        rank_offset: Rank offset for inference ranks in the NCCL group.
        world_size: Total world size of the NCCL weight-update group.
        backend: Communication backend ("nccl", "hccl", "gloo").
        extra: Backend-specific options.
    """

    method: WeightSyncMethod = WeightSyncMethod.NCCL
    checkpoint_dir: str = "/tmp/forge/weight_sync"
    group_name: str = "forge_weight_sync"
    master_addr: str = ""
    master_port: int = 29600
    rank_offset: int = 0
    world_size: int = 0
    backend: str = "nccl"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = WeightSyncMethod(self.method)


@runtime_checkable
class WeightSyncStrategy(Protocol):
    """Protocol for weight synchronization between Trainer and Generator.

    Implementations handle the mechanics of transferring model weights
    from the training process to the inference engine, abstracting away
    the communication layer (NCCL, filesystem, HIXL, etc.).

    Lifecycle::

        strategy = NCCLWeightSync()
        await strategy.initialize(trainer, generator, config)
        # After each training step:
        status = await strategy.push(version=step + 1)
        # Cleanup:
        await strategy.shutdown()
    """

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        config: WeightSyncConfig,
    ) -> dict:
        """Set up the sync channel.

        For NCCL: create process group between trainer and generator ranks.
        For checkpoint: ensure shared directory exists.
        For HIXL: initialize one-sided communication channel.

        Returns metadata about the initialized channel.
        """
        ...

    async def push(self, version: int) -> dict:
        """Push updated weights from trainer to generator.

        Args:
            version: The policy version number after this update.

        Returns:
            Status dict with at least ``{"version": int, "success": bool}``.
        """
        ...

    async def get_status(self) -> dict:
        """Return current sync status (version, timing, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Tear down the sync channel and release resources."""
        ...
