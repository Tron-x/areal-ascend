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

from typing import Any, Protocol, runtime_checkable

# ======================================================================
# New clean protocols (framework-agnostic)
# ======================================================================


@runtime_checkable
class TrainEngine(Protocol):
    """Pluggable training engine for policy optimization.

    Implementations wrap a concrete training framework (TorchTitan,
    AReaL/FSDPEngine, Megatron, etc.) and expose a uniform interface.

    Used by ``TrainerActor`` and the orchestration layer in ``apps/``.
    """

    def initialize(self) -> dict:
        """Initialize the engine. Returns metadata (max_steps, start_step, ...)."""
        ...

    def train_step(self, batch: Any, step: int) -> dict:
        """Run one training step on the given batch. Returns result dict."""
        ...

    def push_weights(self, version: int) -> None:
        """Make updated weights available for inference engines to pull."""
        ...

    def shutdown(self) -> None:
        """Release resources."""
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
