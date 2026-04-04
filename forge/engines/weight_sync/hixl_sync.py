"""HIXLWeightSync — one-sided RDMA weight transfer via Monarch HIXL.

HIXL (Huawei Interconnect eXchange Library) provides one-sided
put/get operations for high-performance weight transfer on NPU clusters.
This is the Ascend-native counterpart to NCCL's collective broadcast.

Status: **Stub** — the protocol and lifecycle are defined, but the
actual HIXL channel setup requires Monarch's HIXL bindings which are
not yet available in the open-source Forge codebase.

When HIXL is available, the flow is:
    1. ``initialize``: Create HIXL channel between trainer and generator.
    2. ``push``: Trainer does one-sided put of state dict shards.
    3. Generator detects version bump and loads from local HIXL buffer.
"""

from __future__ import annotations

import logging
from typing import Any

from forge.core.weight_sync import WeightSyncConfig

logger = logging.getLogger("HIXLWeightSync")


class HIXLWeightSync:
    """Weight sync via Monarch HIXL one-sided communication.

    Requires HIXL bindings in the Monarch runtime.  Falls back to
    a clear error if HIXL is not available.
    """

    def __init__(self) -> None:
        self._trainer_actor: Any = None
        self._generator_actor: Any = None
        self._config: WeightSyncConfig | None = None
        self._initialized = False
        self._current_version = 0

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        config: WeightSyncConfig,
    ) -> dict:
        self._trainer_actor = trainer_actor
        self._generator_actor = generator_actor
        self._config = config

        try:
            import monarch.hixl  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "HIXL weight sync requires monarch.hixl bindings. "
                "Ensure Monarch is built with HIXL support for Ascend NPU."
            )

        self._initialized = True
        logger.info("HIXLWeightSync initialized (stub — full impl pending)")
        return {"status": "initialized", "method": "hixl"}

    async def push(self, version: int) -> dict:
        if not self._initialized:
            raise RuntimeError("HIXLWeightSync not initialized")

        result = await self._generator_actor.update_weights_sync.call(
            version, "hixl", {}
        )
        _, sync_result = next(iter(result.items()))

        if sync_result.get("success"):
            self._current_version = version
            logger.info("HIXL weight push complete: v%d", version)
        else:
            logger.error("HIXL weight push failed: %s", sync_result)

        return {
            "version": version,
            "success": sync_result.get("success", False),
            "message": sync_result.get("message", ""),
        }

    async def get_status(self) -> dict:
        return {
            "method": "hixl",
            "initialized": self._initialized,
            "current_version": self._current_version,
        }

    async def shutdown(self) -> None:
        logger.info("HIXLWeightSync shutdown")
        self._initialized = False
