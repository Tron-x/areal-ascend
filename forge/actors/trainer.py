"""TrainerActor — framework-agnostic training actor for Monarch orchestration.

Depends ONLY on ``forge.core.protocols.TrainBackend``. All framework-specific
logic (AReaL's PPOTrainer, FSDPEngine, perf_tracer, etc.) lives in the
adapter that implements ``TrainBackend`` (e.g. ``forge.adapters.areal``).

Supports two orchestration modes:

- **Synchronous**: ``train_step(step)`` does rollout + train in one call.
- **Async pipeline**: ``train_on_buffered_batch`` + ``sync_weights`` are
  called separately, allowing rollout and training to run in parallel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

if TYPE_CHECKING:
    from forge.core.protocols import TrainBackend

logger = logging.getLogger("TrainerActor")


class TrainerActor(ForgeActor):
    """FSDP/Megatron training actor with pluggable backend.

    The actor is a thin Monarch endpoint wrapper — it delegates all
    training logic to the ``TrainBackend`` instance passed at construction.

    Synchronous lifecycle::

        info = await actor.initialize.call()
        result = await actor.train_step.call(step)   # rollout + train

    Async pipeline lifecycle::

        info = await actor.initialize.call()
        result = await actor.train_on_buffered_batch.call(batch, step)
        await actor.sync_weights.call(step)
    """

    procs = 1
    with_gpus = True

    def __init__(self, backend: TrainBackend, **kwargs: Any):
        self._backend = backend
        self._extra_kwargs = kwargs

    @endpoint
    def initialize(self) -> dict:
        """Initialize the training backend and return metadata."""
        return self._backend.initialize()

    @endpoint
    def train_step(self, global_step: int) -> dict:
        """Combined rollout + training in one step (synchronous mode)."""
        return self._backend.train_step(global_step)

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        """Run rollout only, return serialized batch."""
        return self._backend.do_rollout(global_step)

    @endpoint
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        """Train on a pre-produced rollout batch."""
        return self._backend.train_on_batch(batch_data, global_step)

    @endpoint
    def train_on_buffered_batch(
        self,
        batch_data: dict,
        global_step: int,
        skip_weight_sync: bool = False,
    ) -> dict:
        """Train on a batch from ReplayBuffer (async pipeline mode).

        When ``skip_weight_sync=True``, call ``sync_weights`` separately
        to push updated weights to Generator.
        """
        return self._backend.train_on_buffered_batch(
            batch_data, global_step, skip_weight_sync=skip_weight_sync
        )

    @endpoint
    def sync_weights(self, global_step: int) -> dict:
        """Push updated weights to Generator (async pipeline mode)."""
        return self._backend.sync_weights(global_step)

    @endpoint
    def get_train_metadata(self) -> dict:
        """Return training metadata for the orchestrator."""
        return self._backend.get_train_metadata()

    @endpoint
    def shutdown(self) -> None:
        """Shut down the training backend."""
        self._backend.shutdown()
