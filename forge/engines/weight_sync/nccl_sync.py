"""NCCLWeightSync — broadcast weights via NCCL/HCCL collective.

Extracted from the existing AReaL XCCL weight-update flow:

1. ``initialize``: Create a cross-mesh process group between trainer ranks
   and Generator (vLLM) ranks.  The Generator side calls
   ``/forge/weights/init_group`` (aliased ``/areal_init_weights_update_group``).
2. ``push``: Pause generation → set weight metadata → broadcast via XCCL →
   resume generation.  Maps to the ``_set_weight_meta`` + ``_update_weights_xccl``
   endpoints.

This implementation communicates with the Generator via its Monarch
endpoints, keeping the strategy agnostic to vLLM internals.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any

from forge.core.weight_sync import WeightSyncConfig, WeightsSpec

logger = logging.getLogger("NCCLWeightSync")


class NCCLWeightSync:
    """Weight sync via NCCL/HCCL collective broadcast.

    Best for co-located training and inference on the same cluster
    where a shared process group can be established.
    """

    def __init__(self) -> None:
        self._trainer_actor: Any = None
        self._generator_actor: Any = None
        self._config: WeightSyncConfig | None = None
        self._initialized = False
        self._current_version = 0
        self._group_name = ""

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        config: WeightSyncConfig,
    ) -> dict:
        """Set up NCCL process group between trainer and generator.

        Calls the Generator's ``/forge/weights/init_group`` endpoint to
        join inference ranks into the collective, then sends weight metadata
        so the first ``push()`` can proceed without a separate meta call.
        """
        self._trainer_actor = trainer_actor
        self._generator_actor = generator_actor
        self._config = config
        self._group_name = config.group_name

        master_addr = config.master_addr or _resolve_master_addr()
        master_port = config.master_port

        init_payload = {
            "master_address": master_addr,
            "master_port": master_port,
            "rank_offset": config.rank_offset,
            "world_size": config.world_size,
            "backend": config.backend,
            "group_name": config.group_name,
        }

        logger.info(
            "Initializing NCCL weight sync: addr=%s:%d world=%d backend=%s",
            master_addr,
            master_port,
            config.world_size,
            config.backend,
        )

        result = await generator_actor.handle_request.call(
            "/forge/weights/init_group", init_payload
        )
        _, init_result = next(iter(result.items()))

        if not init_result.get("success"):
            raise RuntimeError(
                f"Failed to initialize NCCL weight sync group: {init_result}"
            )

        self._initialized = True
        logger.info("NCCL weight sync group initialized: %s", self._group_name)
        return {"status": "initialized", "group_name": self._group_name}

    async def set_weight_meta(self, spec: WeightsSpec) -> dict:
        """Send weight metadata to Generator so it knows what to expect.

        Must be called at least once before the first ``push()``, and
        again whenever the model architecture changes (e.g. LoRA swap).
        """
        if not self._initialized:
            raise RuntimeError("NCCLWeightSync not initialized")

        meta_payload = {
            "names": spec.param_names,
            "dtypes": spec.param_dtypes,
            "shapes": [list(s) for s in spec.param_shapes],
            "group_name": self._group_name,
        }

        result = await self._generator_actor.handle_request.call(
            "/forge/weights/set_meta", meta_payload
        )
        _, meta_result = next(iter(result.items()))
        return meta_result

    async def push(self, version: int) -> dict:
        """Broadcast updated weights from trainer to generator via NCCL.

        Pauses generation, performs XCCL broadcast, then resumes.
        """
        if not self._initialized:
            raise RuntimeError("NCCLWeightSync not initialized")

        result = await self._generator_actor.update_weights_sync.call(
            version, "nccl"
        )
        _, sync_result = next(iter(result.items()))

        if sync_result.get("success"):
            self._current_version = version
            logger.info("NCCL weight push complete: v%d", version)
        else:
            logger.error("NCCL weight push failed: %s", sync_result)

        return {
            "version": version,
            "success": sync_result.get("success", False),
            "message": sync_result.get("message", ""),
        }

    async def get_status(self) -> dict:
        return {
            "method": "nccl",
            "initialized": self._initialized,
            "current_version": self._current_version,
            "group_name": self._group_name,
        }

    async def shutdown(self) -> None:
        logger.info("NCCLWeightSync shutdown (group: %s)", self._group_name)
        self._initialized = False


def _resolve_master_addr() -> str:
    """Resolve master address from env or hostname."""
    addr = os.environ.get("MASTER_ADDR", "")
    if not addr:
        addr = socket.gethostbyname(socket.gethostname())
    return addr
