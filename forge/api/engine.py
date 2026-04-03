"""Engine protocols: the central decoupling point for Forge.

Any inference or training backend that satisfies these protocols can be used
with Forge's rollout functions, apps, and pipeline orchestration.

TorchForge has a ``Trainer`` protocol but no ``GenerateEngine``.
Slime has neither -- it uses raw HTTP + Ray remotes.
Forge provides both for clean framework-agnostic composition.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from forge.api.types import GenerateResult, Metrics, SamplingParams, TrainBatch


@runtime_checkable
class GenerateEngine(Protocol):
    """Protocol for text generation backends.

    Implementations may wrap vLLM (in-process or Monarch-based),
    SGLang (HTTP), or any other inference server.
    """

    async def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerateResult]:
        """Generate completions for a batch of prompts.

        Parameters
        ----------
        prompts
            Input text prompts.
        params
            Sampling parameters controlling generation.

        Returns
        -------
        list[GenerateResult]
            One result per prompt, in the same order.
        """
        ...

    async def update_weights(self, version: int) -> None:
        """Load updated model weights (e.g. after a training step).

        Parameters
        ----------
        version
            Monotonically increasing weight version identifier.
        """
        ...

    async def shutdown(self) -> None:
        """Release resources held by the engine."""
        ...


@runtime_checkable
class TrainEngine(Protocol):
    """Protocol for training backends.

    Implementations may wrap AReaL's PPOTrainer + FSDPEngine,
    Slime's MegatronTrainRayActor, TorchForge's TitanTrainer, etc.
    """

    async def train_step(self, batch: TrainBatch) -> Metrics:
        """Run one training step on the given batch.

        Parameters
        ----------
        batch
            A :class:`TrainBatch` with tensors already on the correct device.

        Returns
        -------
        Metrics
            Training metrics for this step (loss, grad_norm, etc.).
        """
        ...

    async def push_weights(self) -> int:
        """Push updated weights to the weight store and return the new version.

        Returns
        -------
        int
            The new weight version number.
        """
        ...

    async def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint to disk.

        Parameters
        ----------
        path
            Directory where the checkpoint should be written.
        """
        ...

    async def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint from disk.

        Parameters
        ----------
        path
            Directory containing the checkpoint.
        """
        ...

    async def shutdown(self) -> None:
        """Release resources held by the training engine."""
        ...
