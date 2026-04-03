"""Advantage computation algorithms.

Framework-agnostic -- operates on ``Episode`` / ``Group`` types from
``forge.core.types``.  No torch dependency at module level so that
this can be tested on CPU without GPU.
"""

from __future__ import annotations

from forge.core.types import Group


def compute_advantages_grpo(group: Group) -> list[float]:
    """GRPO group-relative advantage: A_i = R_i - mean(R).

    No standard deviation normalization (DR-GRPO style) to avoid
    difficulty bias.

    Args:
        group: List of episodes for the same prompt.

    Returns:
        List of advantage values, one per episode.
    """
    if not group:
        return []
    rewards = [e.reward for e in group]
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


def compute_advantages_grpo_normalized(group: Group) -> list[float]:
    """Classic GRPO advantage with std normalization.

    Args:
        group: List of episodes for the same prompt.

    Returns:
        List of advantage values.
    """
    if not group:
        return []
    rewards = [e.reward for e in group]
    n = len(rewards)
    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / max(n - 1, 1)
    std_r = var_r**0.5
    eps = 1e-4
    return [(r - mean_r) / (std_r + eps) for r in rewards]
