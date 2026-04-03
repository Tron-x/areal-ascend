"""Backend protocols — structural contracts for pluggable training/inference/reward.

All protocols use ``typing.Protocol`` with ``@runtime_checkable`` so that any
class implementing the right methods is accepted without explicit inheritance.

Framework adapters (e.g. ``forge.adapters.areal``) implement these protocols,
and forge actors depend ONLY on these protocols — never on concrete backend
classes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainBackend(Protocol):
    """Backend plugged into ``TrainerActor`` for training orchestration.

    A concrete implementation wraps a specific training framework
    (AReaL/PPOTrainer, Slime/Megatron, etc.) and exposes a uniform
    interface that ``TrainerActor`` calls via Monarch endpoints.

    Example (AReaL adapter)::

        class AReaLTrainBackend:
            def initialize(self) -> dict:
                # Load experiment script, capture PPOTrainer, return metadata
                ...
            def train_step(self, global_step: int) -> dict:
                # Combined rollout + train via PPOTrainer internals
                ...
    """

    def initialize(self) -> dict:
        """Initialize the training pipeline.

        Returns:
            Metadata dict with at least:
            - ``max_steps``: total training steps
            - ``start_step``: step to resume from (0 if fresh run)
            - ``steps_per_epoch``: steps in one epoch
        """
        ...

    def train_step(self, global_step: int) -> dict:
        """Combined rollout + training in one step.

        Returns:
            Result dict with ``global_step``, ``epoch``, ``epoch_step``.
        """
        ...

    def do_rollout(self, global_step: int) -> dict:
        """Run rollout only, return serialized batch data."""
        ...

    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        """Train on a pre-produced rollout batch."""
        ...

    def train_on_buffered_batch(
        self, batch_data: dict, global_step: int, skip_weight_sync: bool = False
    ) -> dict:
        """Train on a batch from ReplayBuffer, optionally deferring weight sync."""
        ...

    def sync_weights(self, global_step: int) -> dict:
        """Push updated weights to Generator independently of training."""
        ...

    def get_train_metadata(self) -> dict:
        """Return training metadata (max_steps, steps_per_epoch, etc.)."""
        ...

    def shutdown(self) -> None:
        """Release resources (models, process groups, etc.)."""
        ...


@runtime_checkable
class InferenceBridge(Protocol):
    """Bridge connecting a training backend to inference actors.

    Used internally by ``TrainBackend`` implementations to route
    generation requests to ``GeneratorActor`` via Monarch RPC.
    This replaces the direct dependency on ``areal.api.InferenceEngine``.

    The bridge is constructed by the adapter and injected into the
    training backend — the ``TrainerActor`` itself never touches it.
    """

    def initialize(self, **kwargs: Any) -> None:
        """Set up the bridge (workflow executor, health checks, etc.)."""
        ...

    def destroy(self) -> None:
        """Tear down resources."""
        ...

    async def agenerate(self, request: Any) -> Any:
        """Async generation request routed to the inference actor."""
        ...

    def set_version(self, version: int) -> None:
        """Update the policy weight version."""
        ...

    def get_version(self) -> int:
        """Return the current weight version."""
        ...

    def pause(self) -> None:
        """Pause generation (for weight sync)."""
        ...

    def resume(self) -> None:
        """Resume generation after weight sync."""
        ...


@runtime_checkable
class RewardBackend(Protocol):
    """Backend plugged into ``RewardActor`` for reward computation.

    A concrete implementation loads a reward function and computes
    scalar rewards for prompt-completion pairs.

    Example::

        class AReaLRewardBackend:
            def setup(self, reward_fn_path="areal.reward.gsm8k.gsm8k_reward_fn"):
                self._fn = import_from_string(reward_fn_path)
            def compute_reward(self, prompt, completion, ...):
                return float(self._fn(prompt, completion, ...))
    """

    def setup(self, reward_fn_path: str = "") -> dict:
        """Load the reward function. Returns status dict."""
        ...

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float:
        """Compute reward for a single prompt-completion pair."""
        ...

    def compute_rewards_batch(self, items: list[dict]) -> list[float]:
        """Compute rewards for a batch of items."""
        ...

    def get_stats(self) -> dict:
        """Return backend statistics (call count, timing, etc.)."""
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Provides data batches for rollout, decoupled from the training pipeline.

    Allows the rollout producer to iterate over training data independently
    of the ``TrainerActor``, enabling true parallel rollout and training.
    """

    def get_batch(self) -> list[dict]:
        """Return the next batch of raw data items.

        Each item is a dict with keys like ``prompt``, ``answer``,
        ``messages``, etc. -- the format consumed by rollout workflows.

        Raises ``StopIteration`` when the epoch is exhausted.
        """
        ...

    def reset(self) -> None:
        """Reset the iterator to the beginning of the dataset."""
        ...

    def __len__(self) -> int:
        """Total number of batches per epoch."""
        ...
