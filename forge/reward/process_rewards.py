"""Built-in process-scope rewards for agentic RL.

These demonstrate the per-turn reward path introduced in Phase A3.
Each function takes a :class:`forge.core.trajectory.Trajectory` and
returns a ``list[float]`` the same length as ``trajectory.turns``.

Adding a new process reward is a 3-line change::

    from forge.reward import register_reward

    @register_reward("my_process_signal", scope="process")
    def my_process_signal(trajectory):
        return [0.1 if t.role.value == "assistant" else 0.0
                for t in trajectory.turns]

Auto-discovery will pick it up at ``ensure_loaded`` time; no config
changes required beyond listing the name in the pipeline's YAML.
"""

from __future__ import annotations

from forge.core.trajectory import Trajectory, TurnRole
from forge.reward import register_reward


@register_reward("tool_call_valid", scope="process")
def tool_call_valid_reward(trajectory: Trajectory) -> list[float]:
    """+0.1 for every assistant turn that emitted at least one tool
    call AND at least one successful observation, 0 otherwise.

    Rationale: in ReTool / TIR training loops, we want a small
    positive signal when the agent chose to call a tool *and* the
    tool actually returned a non-error observation.  This keeps the
    agent willing to use tools without overriding the eventual
    final-answer reward.

    Non-assistant turns (``user`` / ``tool`` / ``system``)
    contribute zero so the vector length always equals
    ``len(trajectory.turns)``.
    """
    rewards: list[float] = []
    for turn in trajectory.turns:
        if turn.role != TurnRole.ASSISTANT:
            rewards.append(0.0)
            continue
        if not turn.tool_calls:
            rewards.append(0.0)
            continue
        # Require at least one successful observation on this turn;
        # failed tool calls give 0 so we don't reinforce
        # hallucinated / malformed tool invocations.
        if any(obs.success for obs in turn.observations):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


@register_reward("turn_efficiency", scope="process")
def turn_efficiency_reward(trajectory: Trajectory) -> list[float]:
    """Tiny negative signal (-0.01) for every assistant turn so the
    agent prefers shorter trajectories when final reward is tied.

    Disabled by default (weight=0 in most pipelines).  Included as a
    canonical example of a *cost*-style process reward -- the
    pipeline handles negative values identically to positive ones,
    no special casing needed.
    """
    return [
        -0.01 if turn.role == TurnRole.ASSISTANT else 0.0 for turn in trajectory.turns
    ]


__all__ = ["tool_call_valid_reward", "turn_efficiency_reward"]
