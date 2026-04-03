"""TrainerActor — framework-agnostic training actor for Monarch orchestration.

Depends ONLY on ``forge.core.protocols.TrainBackend``. All framework-specific
logic (AReaL's PPOTrainer, FSDPEngine, perf_tracer, etc.) lives in the
adapter that implements ``TrainBackend`` (e.g. ``forge.adapters.areal``).
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

    Lifecycle::

        backend = AReaLTrainBackend(cli_args=[...], ...)
        actor = await TrainerActor.options(procs=4, with_gpus=True).as_actor(
            backend=backend,
        )
        info = await actor.initialize.call()
        result = await actor.train_step.call(step)
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
        """Combined rollout + training in one step."""
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
    def shutdown(self) -> None:
        """Shut down the training backend."""
        self._backend.shutdown()
