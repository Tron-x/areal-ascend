"""Framework-agnostic RL algorithms for Forge.

Provides advantage computation, loss functions, and batch collation
that work with any training engine backend.
"""

from forge.rl.advantage import compute_advantages_grpo
from forge.rl.collate import collate_episodes

__all__ = [
    "collate_episodes",
    "compute_advantages_grpo",
]
