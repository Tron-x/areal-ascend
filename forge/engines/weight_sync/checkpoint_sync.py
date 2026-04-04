"""CheckpointWeightSync — save-to-disk weight transfer.

The trainer saves its state dict to a shared filesystem path, and the
Generator reloads weights from that path.  This is the simplest and
most robust strategy, suitable for:

- Cross-cluster deployments (training and inference on different clusters).
- Elastic scheduling where the process group topology is unstable.
- Debugging and offline analysis of intermediate checkpoints.

Flow:
    1. ``initialize``: Ensure the shared checkpoint directory exists.
    2. ``push``: Trainer saves state_dict → Generator calls ``_update_weights_disk``.
    3. Generator reloads from the checkpoint path.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from forge.core.weight_sync import WeightSyncConfig

logger = logging.getLogger("CheckpointWeightSync")


class CheckpointWeightSync:
    """Weight sync via shared filesystem checkpoint.

    The trainer writes a checkpoint to ``checkpoint_dir/v{version}/``,
    and the generator reloads from the same path.  Atomic rename is
    used (write to temp, rename) to avoid partial-read issues.
    """

    def __init__(self) -> None:
        self._trainer_actor: Any = None
        self._generator_actor: Any = None
        self._config: WeightSyncConfig | None = None
        self._checkpoint_dir = ""
        self._initialized = False
        self._current_version = 0
        self._push_times: list[float] = []

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        config: WeightSyncConfig,
    ) -> dict:
        """Ensure the shared checkpoint directory exists."""
        self._trainer_actor = trainer_actor
        self._generator_actor = generator_actor
        self._config = config
        self._checkpoint_dir = config.checkpoint_dir

        os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._initialized = True
        logger.info(
            "CheckpointWeightSync initialized: dir=%s", self._checkpoint_dir
        )
        return {"status": "initialized", "checkpoint_dir": self._checkpoint_dir}

    async def push(self, version: int) -> dict:
        """Save trainer weights to checkpoint and reload on generator.

        Steps:
            1. Request state dict from trainer via ``state_dict_for_sync``.
            2. Save to ``{checkpoint_dir}/v{version}/``.
            3. Tell generator to reload from that path.
        """
        if not self._initialized:
            raise RuntimeError("CheckpointWeightSync not initialized")

        t0 = time.monotonic()

        version_dir = os.path.join(self._checkpoint_dir, f"v{version}")
        os.makedirs(version_dir, exist_ok=True)

        state_dict_result = await self._trainer_actor.state_dict_for_sync.call()
        _, state_dict = next(iter(state_dict_result.items()))

        self._save_state_dict(state_dict, version_dir)

        result = await self._generator_actor.update_weights_sync.call(
            version, "checkpoint", {"model_path": version_dir}
        )
        _, sync_result = next(iter(result.items()))

        elapsed = time.monotonic() - t0
        self._push_times.append(elapsed)

        if sync_result.get("success"):
            self._current_version = version
            logger.info(
                "Checkpoint weight push complete: v%d (%.2fs)", version, elapsed
            )
            self._cleanup_old_checkpoints(version)
        else:
            logger.error("Checkpoint weight push failed: %s", sync_result)

        return {
            "version": version,
            "success": sync_result.get("success", False),
            "elapsed_s": elapsed,
            "checkpoint_path": version_dir,
        }

    async def get_status(self) -> dict:
        avg_time = (
            sum(self._push_times) / len(self._push_times)
            if self._push_times
            else 0.0
        )
        return {
            "method": "checkpoint",
            "initialized": self._initialized,
            "current_version": self._current_version,
            "checkpoint_dir": self._checkpoint_dir,
            "avg_push_time_s": avg_time,
            "total_pushes": len(self._push_times),
        }

    async def shutdown(self) -> None:
        logger.info("CheckpointWeightSync shutdown")
        self._initialized = False

    def _save_state_dict(self, state_dict: dict, path: str) -> None:
        """Save state dict to disk using torch.save with atomic write."""
        import torch

        tmp_path = path + ".tmp"
        os.makedirs(tmp_path, exist_ok=True)
        save_path = os.path.join(tmp_path, "model.pt")
        torch.save(state_dict, save_path)

        final_path = os.path.join(path, "model.pt")
        os.replace(save_path, final_path)

        try:
            os.rmdir(tmp_path)
        except OSError:
            pass

        logger.debug("Saved state dict to %s", final_path)

    def _cleanup_old_checkpoints(self, current_version: int, keep: int = 2) -> None:
        """Remove old checkpoint versions, keeping the most recent ``keep``."""
        import shutil

        if not self._checkpoint_dir:
            return

        try:
            versions = []
            for name in os.listdir(self._checkpoint_dir):
                if name.startswith("v") and name[1:].isdigit():
                    versions.append(int(name[1:]))
            versions.sort()

            to_remove = versions[:-keep] if len(versions) > keep else []
            for v in to_remove:
                vdir = os.path.join(self._checkpoint_dir, f"v{v}")
                shutil.rmtree(vdir, ignore_errors=True)
                logger.debug("Cleaned up old checkpoint: %s", vdir)
        except Exception as e:
            logger.warning("Checkpoint cleanup failed: %s", e)
