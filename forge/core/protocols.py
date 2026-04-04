"""Engine protocols -- pluggable contracts for training, inference, and reward.

Two protocol levels:

1. **Engine protocols** (new, clean): ``TrainEngine``, ``InferenceEngine``,
   ``RewardFn`` -- simple interfaces that any backend can implement.
   These are what new code should target.

2. **Legacy backend protocols**: ``TrainBackend``, ``InferenceBridge``,
   ``RewardBackend``, ``DataProvider`` -- retained for backward compatibility
   with ``forge/engines/areal/``.  They add AReaL-specific methods like
   ``do_rollout``, ``train_on_batch``, etc.

All protocols use ``typing.Protocol`` with ``@runtime_checkable``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from forge.core.weight_sync import WeightsSpec

# ======================================================================
# New clean protocols (framework-agnostic)
# ======================================================================


@runtime_checkable
class TrainEngine(Protocol):
    """Framework-agnostic training engine for policy optimization.

    Implementations wrap a concrete training framework (FSDP2, Megatron,
    AReaL PPOTrainer, etc.) and expose a uniform interface that the
    orchestration layer (``TrainerActor``, ``apps/``) can drive without
    knowledge of the underlying framework.

    Key design decisions:

    - ``train_step(batch, step)`` receives an **externally-provided** batch,
      decoupling data production from training.
    - ``get_weights_spec()`` + ``state_dict_for_sync()`` separate weight
      *description* from weight *data*, letting ``WeightSyncStrategy``
      decide how to transfer weights.
    - Weight pushing and rollout are **not** part of this protocol --
      they belong to the orchestration layer.

    Lifecycle::

        engine = FSDPTrainEngine(config)
        meta = engine.initialize()        # load model, optimizer, scheduler
        for step in range(meta["max_steps"]):
            result = engine.train_step(batch, step)
            # Orchestrator handles weight sync separately
        engine.shutdown()
    """

    def initialize(self) -> dict:
        """Load model, optimizer, and scheduler.

        Returns:
            Metadata dict with at least:
            ``{"max_steps": int, "start_step": int, "model_path": str}``.
        """
        ...

    def train_step(self, batch: dict, step: int) -> dict:
        """Run one optimization step on an externally-provided batch.

        Args:
            batch: Engine-specific tensor dict produced by a ``BatchAdapter``.
            step: Global training step number.

        Returns:
            Result dict with at least ``{"loss": float}``.
            May also include ``"grad_norm"``, ``"lr"``, etc.
        """
        ...

    def get_weights_spec(self) -> WeightsSpec:
        """Describe the current model parameters for weight sync.

        Returns a ``WeightsSpec`` containing parameter names, shapes,
        dtypes, and optional sharding metadata -- everything a
        ``WeightSyncStrategy`` needs to set up a transfer channel.
        """
        ...

    def state_dict_for_sync(self) -> dict:
        """Return a state dict (or shard) for weight sync to inference.

        The returned dict maps parameter names to tensors. For sharded
        models (FSDP, Megatron), this may return only the local shard;
        the ``WeightSyncStrategy`` handles reassembly.
        """
        ...

    def get_metadata(self) -> dict:
        """Return engine metadata.

        Returns:
            Dict with ``"max_steps"``, ``"current_step"``,
            ``"model_path"``, and any engine-specific info.
        """
        ...

    def shutdown(self) -> None:
        """Release resources (model, optimizer, CUDA memory)."""
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Pluggable inference engine for text generation.

    Implementations wrap vLLM, SGLang, TGI, etc.

    Used by ``Generator`` actor.
    """

    async def generate(self, prompt: str, **kwargs: Any) -> dict:
        """Generate text for a prompt. Returns completion dict."""
        ...

    def update_weights(self, version: int) -> None:
        """Pull updated weights from the training engine."""
        ...

    def get_version(self) -> int:
        """Current policy version."""
        ...


@runtime_checkable
class RewardFn(Protocol):
    """Pluggable reward function.

    Can be a simple callable or a full model-based reward.
    """

    def __call__(
        self, prompt: str, response: str, target: Any = None, **kwargs: Any
    ) -> float:
        """Compute scalar reward for a prompt-response pair."""
        ...


@runtime_checkable
class RewardModelEngine(Protocol):
    """Pluggable neural reward model for scoring (prompt, response) pairs.

    Unlike ``RewardFn`` (a stateless callable), a ``RewardModelEngine``
    manages a loaded model with GPU memory, batched inference, and
    optional weight updates.

    Lifecycle::

        engine = HFRewardModelEngine(model_path="...")
        engine.load()                          # load checkpoint to GPU
        scores = engine.score_batch([...])     # batched inference
        engine.shutdown()                      # free GPU memory

    Implementations live in ``forge/engines/<backend>/reward_model.py``.
    """

    def load(self) -> dict:
        """Load model checkpoint and move to device. Returns metadata."""
        ...

    def score(self, prompt: str, response: str) -> float:
        """Score a single (prompt, response) pair. Returns scalar reward."""
        ...

    def score_batch(self, items: list[dict[str, str]]) -> list[float]:
        """Score a batch of (prompt, response) pairs.

        Each item dict must contain ``"prompt"`` and ``"response"`` keys.
        Returns list of scalar rewards.
        """
        ...

    def shutdown(self) -> None:
        """Free model and GPU memory."""
        ...


@runtime_checkable
class BatchAdapter(Protocol):
    """Convert framework-agnostic ``Episode`` objects to engine-specific batches.

    Each training engine (AReaL, TorchTitan, Slime, ...) expects a different
    tensor layout.  A ``BatchAdapter`` bridges the gap so that actors
    produce only ``Episode`` objects and the orchestrator converts them
    at the last moment before sending to the ``TrainerActor``.

    Implementations live in ``forge/engines/<backend>/batch_adapter.py``.
    """

    def adapt(self, episodes: list) -> dict:
        """Convert a list of Episodes to the engine's expected batch dict.

        Args:
            episodes: Framework-agnostic ``Episode`` objects from the
                rollout pipeline.

        Returns:
            A dict of lists/tensors that the training engine can consume
            directly (e.g. ``input_ids``, ``attention_mask``, etc.).
        """
        ...

    def required_fields(self) -> list[str]:
        """Return the list of Episode fields this adapter needs.

        Used for validation: the orchestrator can warn early if an
        Episode is missing a required field.
        """
        ...


@runtime_checkable
class AgentLogic(Protocol):
    """Pluggable agent strategy -- pure logic, no infrastructure awareness.

    Implementations define *what* the agent does (extract tools, decide
    when to stop, format feedback) while ``AgentActor`` handles *how*
    (call Generator via ModelProxy, execute tools, collect training data).

    Built-in implementations:
        - ``forge.agents.react.SimpleReActAgent``   (code-execution ReAct loop)
    """

    def process_response(
        self,
        response: str,
        messages: list[dict[str, str]],
    ) -> Any:
        """Analyse a generation response and decide what to do.

        Returns an ``AgentAction`` describing extracted tool calls and
        whether the episode is considered finished.
        """
        ...

    def should_continue(self, turn: int, reward: float) -> bool:
        """Decide whether to proceed to the next turn."""
        ...

    def format_feedback(
        self,
        action: Any,
        tool_results: list,
        reward: float,
    ) -> str:
        """Build the user-feedback message appended before the next turn."""
        ...


# ======================================================================
# Legacy backend protocols (backward compat for engines/areal)
# ======================================================================


@runtime_checkable
class TrainBackend(Protocol):
    """Legacy training backend with AReaL-style rollout+train interface.

    Kept for backward compatibility. New engines should implement
    ``TrainEngine`` instead.
    """

    def initialize(self) -> dict: ...
    def train_step(self, global_step: int) -> dict: ...
    def do_rollout(self, global_step: int) -> dict: ...
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict: ...

    def train_on_buffered_batch(
        self, batch_data: dict, global_step: int, skip_weight_sync: bool = False
    ) -> dict: ...

    def sync_weights(self, global_step: int) -> dict: ...
    def get_train_metadata(self) -> dict: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class InferenceBridge(Protocol):
    """Legacy inference bridge (AReaL-style, used inside TrainBackend)."""

    def initialize(self, **kwargs: Any) -> None: ...
    def destroy(self) -> None: ...
    async def agenerate(self, request: Any) -> Any: ...
    def set_version(self, version: int) -> None: ...
    def get_version(self) -> int: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


@runtime_checkable
class RewardBackend(Protocol):
    """Legacy reward backend (AReaL-style, used by RewardActor)."""

    def setup(self, reward_fn_path: str = "") -> dict: ...

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float: ...

    def compute_rewards_batch(self, items: list[dict]) -> list[float]: ...
    def get_stats(self) -> dict: ...


@runtime_checkable
class DataProvider(Protocol):
    """Data provider for rollout (decoupled from training pipeline)."""

    def get_batch(self) -> list[dict]: ...
    def reset(self) -> None: ...
    def __len__(self) -> int: ...
