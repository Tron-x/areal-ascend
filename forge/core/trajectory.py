"""Multi-turn agentic trajectory -- first-class data type for Phase A3.

:class:`Episode` models a single-turn RL episode (prompt → completion →
scalar reward).  Agentic RL needs more:

* A conversation of multiple turns (user / assistant / tool).
* Tool calls emitted mid-turn and their results.
* Per-turn rewards (process reward) alongside the final reward.

This module provides the minimal data types to express that cleanly,
with zero dependencies outside the stdlib so they flow through Monarch
RPC as plain dicts without pickling torch state.

Design notes:

* ``Trajectory`` is a *new* type -- it does NOT replace ``Episode``.
  Single-turn rollouts stay on ``Episode`` (less ceremony, better cache
  locality for ReplayBuffer).  Multi-turn rollouts produce a
  ``Trajectory`` which can be flattened to an ``Episode`` for the
  training engine via :meth:`Trajectory.to_episode`.
* Per-turn rewards live on the ``Turn`` itself (``Turn.reward``) and
  also aggregate on ``Trajectory.step_rewards``.  The two views are
  kept in sync by :meth:`Trajectory.sync_step_rewards`; callers can
  write to either side.
* ``ToolCall`` / ``ToolResult`` are re-exported from
  :mod:`forge.tools.protocol` to avoid duplication -- trajectories
  reference the same protocol the tool server exposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from forge.tools.protocol import ToolCall, ToolResult

__all__ = [
    "TurnRole",
    "Observation",
    "Turn",
    "Trajectory",
    "ToolCall",
    "ToolResult",
]


class TurnRole(str, Enum):
    """Who produced this turn.

    Uses ``str`` base so JSON / dict round-trips stay plain strings
    (``turn.role == "assistant"`` works) without needing a custom
    serializer.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class Observation:
    """Per-turn environmental / tool observation surfaced back to the agent.

    Separate from ``ToolResult`` because an observation can also be
    produced by a non-tool environment (e.g. web browser DOM, game
    state).  For tool-based turns, ``from_tool_result`` factory wraps
    a :class:`ToolResult` into the right shape.

    Attributes:
        text: Human-readable / prompt-ready text the agent will see.
        success: Whether the underlying action succeeded.
        metadata: Free-form structured payload (tool name, raw dict
            result, latency, ...).
    """

    text: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Observation:
        text = result.output or result.error or ""
        meta: dict[str, Any] = {}
        if result.tool_call is not None:
            meta["tool_name"] = result.tool_call.name
            meta["tool_args"] = result.tool_call.arguments
        if result.error:
            meta["error"] = result.error
        return cls(text=text, success=result.success, metadata=meta)


@dataclass
class Turn:
    """One step of an agentic episode.

    A turn groups: the text the agent saw / produced, any tool calls
    invoked, their results, and the scalar reward attributed to this
    turn by the process-reward pipeline.

    Attributes:
        turn_idx: 0-based index into the trajectory.
        role: Producer of the turn (:class:`TurnRole`).
        content: Text of the turn (rendered prompt for ``user`` /
            response text for ``assistant`` / observation text for
            ``tool``).
        tool_calls: Tool calls emitted by this turn (only populated
            for ``assistant`` turns that invoked tools).
        observations: Observations surfaced to the next turn (only
            populated for ``tool`` turns or environmentally driven
            turns).
        reward: Process reward attributed to this turn.  Defaults to
            0.0 so turns without a scoped reward contribute nothing.
        token_ids: Optional tokenized turn content (training).
        logprobs: Optional per-token generator logprobs (training).
        loss_mask: Optional per-token loss mask; typically 1 for
            assistant-generated tokens and 0 for tool / user tokens.
        metadata: Free-form turn metadata.
    """

    turn_idx: int = 0
    role: TurnRole = TurnRole.ASSISTANT
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    reward: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict serialization for Monarch RPC.  Nested
        dataclasses are flattened so the receiving side doesn't need
        to import :mod:`forge.tools.protocol` before unpickling."""
        return {
            "turn_idx": self.turn_idx,
            "role": self.role.value
            if isinstance(self.role, TurnRole)
            else str(self.role),
            "content": self.content,
            "tool_calls": [
                {"name": c.name, "arguments": c.arguments, "raw_text": c.raw_text}
                for c in self.tool_calls
            ],
            "observations": [
                {"text": o.text, "success": o.success, "metadata": dict(o.metadata)}
                for o in self.observations
            ],
            "reward": self.reward,
            "token_ids": list(self.token_ids),
            "logprobs": list(self.logprobs),
            "loss_mask": list(self.loss_mask),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn:
        role_raw = d.get("role", "assistant")
        try:
            role = TurnRole(role_raw)
        except ValueError:
            role = TurnRole.ASSISTANT
        return cls(
            turn_idx=int(d.get("turn_idx", 0)),
            role=role,
            content=d.get("content", ""),
            tool_calls=[
                ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}) or {},
                    raw_text=tc.get("raw_text", ""),
                )
                for tc in d.get("tool_calls", [])
            ],
            observations=[
                Observation(
                    text=o.get("text", ""),
                    success=bool(o.get("success", True)),
                    metadata=dict(o.get("metadata", {}) or {}),
                )
                for o in d.get("observations", [])
            ],
            reward=float(d.get("reward", 0.0)),
            token_ids=list(d.get("token_ids", [])),
            logprobs=list(d.get("logprobs", [])),
            loss_mask=list(d.get("loss_mask", [])),
            metadata=dict(d.get("metadata", {}) or {}),
        )


@dataclass
class Trajectory:
    """Full multi-turn trajectory produced by an agentic rollout.

    A trajectory is the natural unit of work for Phase A3's reward
    pipeline: final-scope rewards read ``prompt + last assistant turn
    content + target``, process-scope rewards iterate ``turns`` and
    attach a scalar to each.

    Attributes:
        trajectory_id: Unique identifier (usually == episode id).
        prompt: The seed prompt (may be re-expressed inside
            ``turns[0]`` for tokenization; kept separately for
            reward-fn back-compat).
        target: Ground-truth answer / reference signal (for
            final-scope rewards).
        turns: Ordered list of :class:`Turn` making up the episode.
        final_reward: Scalar reward attributed to the trajectory as
            a whole (final-scope rewards contribute here).
        step_rewards: Cached list of per-turn rewards (same length as
            ``turns``).  :meth:`sync_step_rewards` keeps this in
            lockstep with ``turns[i].reward``.
        reward_breakdown: Per-reward-name scalar decomposition (for
            logging / debugging).  Process rewards store the SUM of
            their per-turn values here; the per-turn breakdown is on
            the individual turn's ``metadata['step_reward_breakdown']``.
        policy_version: Policy version that generated the trajectory.
        metadata: Free-form trajectory metadata.
    """

    trajectory_id: str = ""
    prompt: str = ""
    target: Any = None
    turns: list[Turn] = field(default_factory=list)
    final_reward: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    policy_version: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----- convenience helpers ---------------------------------------

    def assistant_turns(self) -> list[Turn]:
        """Return only the assistant-produced turns in order.

        Convenience for process rewards that want to score only model
        outputs (and ignore tool / observation turns).
        """
        return [t for t in self.turns if t.role == TurnRole.ASSISTANT]

    def final_response(self) -> str:
        """Concatenated assistant turn content -- the "answer" that
        final-scope rewards typically compare against ``target``.

        Returns the last assistant turn's content when the trajectory
        has at least one assistant turn; empty string otherwise.
        """
        asst = self.assistant_turns()
        return asst[-1].content if asst else ""

    def sync_step_rewards(self) -> None:
        """Mirror ``turns[i].reward`` into ``self.step_rewards``.

        The two views exist because different consumers prefer each
        form: ``step_rewards`` is trivially serializable and easy to
        feed into the trainer, while ``turns[i].reward`` keeps the
        scalar next to the content that earned it.  Keep them aligned
        after mutations by calling this once.
        """
        self.step_rewards = [t.reward for t in self.turns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "prompt": self.prompt,
            "target": self.target,
            "turns": [t.to_dict() for t in self.turns],
            "final_reward": self.final_reward,
            "step_rewards": list(self.step_rewards),
            "reward_breakdown": dict(self.reward_breakdown),
            "policy_version": self.policy_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        return cls(
            trajectory_id=d.get("trajectory_id", ""),
            prompt=d.get("prompt", ""),
            target=d.get("target"),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            final_reward=float(d.get("final_reward", 0.0)),
            step_rewards=list(d.get("step_rewards", [])),
            reward_breakdown=dict(d.get("reward_breakdown", {}) or {}),
            policy_version=int(d.get("policy_version", -1)),
            metadata=dict(d.get("metadata", {}) or {}),
        )

    @classmethod
    def from_single_turn(
        cls,
        prompt: str,
        response: str,
        reward: float = 0.0,
        target: Any = None,
        trajectory_id: str = "",
        **metadata: Any,
    ) -> Trajectory:
        """Build a trivial single-turn trajectory wrapping a legacy
        ``(prompt, response, reward)`` tuple.

        Useful when an existing single-turn pipeline wants to opt into
        process-reward scoring without first rewriting its episode
        flow; feed this into :class:`RewardPipeline`, then unpack the
        final scalar + breakdown.
        """
        turn = Turn(
            turn_idx=0,
            role=TurnRole.ASSISTANT,
            content=response,
            reward=0.0,  # pipeline fills
        )
        return cls(
            trajectory_id=trajectory_id,
            prompt=prompt,
            target=target,
            turns=[turn],
            final_reward=reward,
            metadata=dict(metadata),
        )
