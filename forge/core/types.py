"""Universal data types for the Forge training pipeline.

These types are the common language between all components (actors, engines,
RL algorithms, orchestration).  They have ZERO framework dependencies --
no AReaL, no Slime, no TorchTitan.  Only stdlib + typing.

Inspired by TorchForge's ``Episode``, ``Completion``, and ``TrainBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Completion:
    """A model-generated completion for a given prompt.

    Attributes:
        text: Decoded generated text.
        prompt: The original prompt string.
        prompt_ids: Encoded prompt token IDs.
        token_ids: Encoded generated token IDs.
        logprobs: Per-token log-probabilities of the generated tokens.
        stop_reason: Why generation stopped (``"stop"``, ``"length"``, etc.).
        generator_version: Policy version that produced this completion.
        metadata: Extra info (timing, model name, etc.).
    """

    text: str = ""
    prompt: str = ""
    prompt_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str | None = None
    generator_version: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A single RL episode: prompt + completion + reward + training signals.

    This is the universal unit of data flowing through the pipeline::

        Generator -> Episode -> ReplayBuffer -> Trainer

    Attributes:
        episode_id: Unique identifier.
        prompt: Original prompt text.
        response: Generated response text.
        target: Ground truth answer (for reward computation).
        completion: Full generation metadata.
        reward: Scalar reward for this episode.
        reward_breakdown: Per-component reward scores.
        advantage: Computed advantage (GRPO/GAE).
        policy_version: Which policy version generated this.
        generator_logprobs: Per-token logprobs from the generator.
        ref_logprobs: Per-token logprobs from the reference model.
        loss_mask: Binary mask indicating which tokens to train on.
        metadata: Arbitrary extra data.
    """

    episode_id: str = ""
    prompt: str = ""
    response: str = ""
    target: Any = None
    completion: Completion | None = None
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    advantage: float | None = None
    policy_version: int = -1
    generator_logprobs: list[float] = field(default_factory=list)
    ref_logprobs: list[float] | None = None
    loss_mask: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainBatch:
    """Universal training batch consumed by any TrainEngine.

    Separates model inputs from loss inputs, so the trainer can do::

        logits = model(**batch.model_inputs)
        loss = loss_fn(logits, **batch.loss_inputs)

    Attributes:
        model_inputs: Inputs for the forward pass (input_ids, attention_mask, ...).
        loss_inputs: Inputs for loss computation (advantages, ref_logprobs, ...).
        meta: Non-training metadata (for logging, checkpointing, etc.).
    """

    model_inputs: dict[str, Any] = field(default_factory=dict)
    loss_inputs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


Group = list[Episode]
"""A group of episodes for the same prompt (GRPO: G completions per prompt)."""


# ======================================================================
# Agent types (moved from core/agent.py for a flatter core/ structure)
# ======================================================================


@dataclass
class ToolCall:
    """A single tool invocation requested by the agent.

    Attributes:
        type: Tool identifier (e.g. ``"code_execution"``, ``"web_search"``).
        content: Payload for the tool (source code, query string, etc.).
        metadata: Optional extra data for the tool.
    """

    type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Outcome of executing a ``ToolCall``.

    Attributes:
        success: Whether the tool executed without errors.
        output: The tool's stdout / return value.
        error: Error message if ``success`` is False.
        tool_call: The original request that produced this result.
    """

    success: bool
    output: str = ""
    error: str = ""
    tool_call: ToolCall | None = None


@dataclass
class GenerationResult:
    """LLM generation output with RL training metadata.

    The ``text`` field is all an ``AgentLogic`` needs.  The remaining
    fields are collected by ``AgentActor`` for training data assembly
    and are opaque to agent logic implementations.

    Attributes:
        text: Generated text.
        token_ids: Output token IDs (for training).
        logprobs: Per-token log-probabilities (for PPO/GRPO advantage).
        version: Generator weight version at generation time.
        raw: Full upstream response dict (preserved for adapters).
    """

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    version: int = -1
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Output of a single agent reasoning step.

    Attributes:
        response: The model's textual response for this turn.
        tool_calls: Tool invocations extracted from the response.
        done: If True, the agent considers the episode finished.
        metadata: Arbitrary data the agent logic wants to carry forward.
    """

    response: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
