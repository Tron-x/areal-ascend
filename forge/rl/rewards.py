"""Reward utilities for agentic RL.

Pure functions -- no framework dependencies, no side effects.

- ``reward_to_go``: convert sparse final reward into dense per-step returns
- ``process_reward``: assign intermediate step-level rewards
- ``composite_reward``: combine multiple reward signals
"""

from __future__ import annotations


def reward_to_go(
    step_rewards: list[float],
    gamma: float = 1.0,
) -> list[float]:
    """Compute discounted return from each step to the end.

    Converts a sparse reward sequence (e.g. ``[0, 0, 0, 1]``) into
    dense per-step returns (e.g. ``[1, 1, 1, 1]`` with gamma=1).

    Critical for long-horizon agentic tasks (200k+ tokens) where
    a single outcome reward creates high gradient variance.

    Reference: MiniMax Forge "Reward-to-go for Variance Reduction".

    Args:
        step_rewards: Per-step reward values. In many cases only the
            last step has a non-zero reward.
        gamma: Discount factor. 1.0 = no discounting (default).

    Returns:
        Per-step returns ``R_t = r_t + gamma * R_{t+1}``.
    """
    n = len(step_rewards)
    if n == 0:
        return []
    returns = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        running = step_rewards[t] + gamma * running
        returns[t] = running
    return returns


def spread_final_reward(
    num_steps: int,
    final_reward: float,
    gamma: float = 1.0,
) -> list[float]:
    """Convert a single final reward into per-step returns.

    Convenience wrapper for the common case where only the last
    step has a reward.

    Args:
        num_steps: Total number of steps in the trajectory.
        final_reward: The outcome reward (e.g. 1.0 for correct).
        gamma: Discount factor.

    Returns:
        Per-step returns.
    """
    step_rewards = [0.0] * num_steps
    if num_steps > 0:
        step_rewards[-1] = final_reward
    return reward_to_go(step_rewards, gamma)


def process_reward(
    step_events: list[dict],
    rules: dict[str, float] | None = None,
) -> list[float]:
    """Assign intermediate step-level rewards based on rules.

    Provides dense feedback on intermediate behaviors rather than
    relying solely on the final outcome.

    Reference: MiniMax Forge "Process Reward: target intermediate
    behaviors (e.g. penalizing language mixing or tool errors)".

    Args:
        step_events: List of event dicts, one per step. Each dict
            may contain keys like ``"tool_success"``, ``"tool_error"``,
            ``"language_switch"``, etc.
        rules: Mapping from event key to reward value.
            Default rules::

                {
                    "tool_success": +0.1,
                    "tool_error": -0.1,
                    "language_switch": -0.05,
                    "answer_found": +0.5,
                }

    Returns:
        Per-step reward values.
    """
    if rules is None:
        rules = {
            "tool_success": 0.1,
            "tool_error": -0.1,
            "language_switch": -0.05,
            "answer_found": 0.5,
        }

    rewards = []
    for event in step_events:
        r = 0.0
        for key, value in rules.items():
            if event.get(key, False):
                r += value
        rewards.append(r)
    return rewards


def composite_reward(
    components: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Combine multiple reward signals into a single scalar.

    Reference: MiniMax Forge "composite reward framework".

    Args:
        components: Named reward values, e.g.
            ``{"correctness": 1.0, "speed": 0.8, "tool_use": 0.5}``.
        weights: Per-component weights. Missing keys default to 1.0.

    Returns:
        Weighted sum of reward components.
    """
    if weights is None:
        weights = {}
    total = 0.0
    for key, value in components.items():
        w = weights.get(key, 1.0)
        total += w * value
    return total
