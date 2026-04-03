"""Framework-agnostic RL algorithms for Forge.

Provides advantage computation, loss functions, and batch collation
that work with any training engine backend.

Loss functions:
    ``GRPOLoss``  -- DR-GRPO (fixed-horizon, asymmetric clip, KL penalty)
    ``DAPOLoss``  -- DAPO (dual-clip for negative advantages)

Primitives:
    ``compute_logprobs``, ``compute_ratio``, ``compute_kl``,
    ``compute_entropy``, ``pg_ppo_clip``, ``aggregate``
"""

from forge.rl.advantage import compute_advantages_grpo
from forge.rl.collate import collate_episodes
from forge.rl.loss import DAPOLoss, GRPOLoss, LossOutput
from forge.rl.rewards import (
    composite_reward,
    process_reward,
    reward_to_go,
    spread_final_reward,
)

__all__ = [
    "DAPOLoss",
    "GRPOLoss",
    "LossOutput",
    "collate_episodes",
    "composite_reward",
    "compute_advantages_grpo",
    "process_reward",
    "reward_to_go",
    "spread_final_reward",
]
