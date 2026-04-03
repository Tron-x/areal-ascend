"""Monarch adapters for Forge's pipeline Protocols.

Thin wrappers that let existing Monarch actors satisfy
``RolloutStage`` and ``TrainStage`` without modifying the actors.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("forge.monarch.stages")


class MonarchRolloutStage:
    """Wraps an AReaL ``RolloutActor`` (Monarch) as ``RolloutStage``."""

    def __init__(self, rollout_actor: Any) -> None:
        self._actor = rollout_actor

    async def produce_batch(self, step: int) -> dict[str, Any]:
        return await self._actor.do_rollout.call_one(step)


class MonarchTrainStage:
    """Wraps an AReaL ``TrainerActor`` (Monarch) as ``TrainStage``.

    Handles single-rank and multi-rank (FSDP) transparently.
    """

    def __init__(self, trainer_actor: Any, *, multi_rank: bool = False) -> None:
        self._actor = trainer_actor
        self._multi_rank = multi_rank
        self._info: dict[str, Any] | None = None

    async def consume_batch(self, batch: dict[str, Any], step: int) -> dict[str, Any]:
        if self._multi_rank:
            result_mesh = await self._actor.train_on_batch.call(batch, step)
            return result_mesh.item(npu=0)
        return await self._actor.train_on_batch.call_one(batch, step)

    async def get_info(self) -> dict[str, Any]:
        if self._info is not None:
            return self._info
        info_mesh = await self._actor.initialize.call()
        _, info = next(iter(info_mesh.items()))
        self._info = info
        return info
