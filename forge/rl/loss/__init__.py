"""RL policy gradient loss functions.

Composable primitives + ready-to-use loss classes:

Primitives (``ops.py``):
    ``compute_logprobs``, ``compute_ratio``, ``compute_kl``,
    ``compute_entropy``, ``pg_ppo_clip``, ``aggregate``

Loss classes:
    ``GRPOLoss``  -- DR-GRPO with asymmetric clip + KL penalty
    ``DAPOLoss``  -- DAPO with dual-clip for negative advantages

Adapted from TorchForge (Meta) with no external dependencies
beyond PyTorch.
"""

from forge.rl.loss.dapo import DAPOLoss
from forge.rl.loss.grpo import GRPOLoss
from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    create_shifted_targets,
    masked_mean,
    pg_ppo_clip,
)
from forge.rl.loss.types import LossOutput

__all__ = [
    "DAPOLoss",
    "GRPOLoss",
    "LossOutput",
    "aggregate",
    "compute_entropy",
    "compute_kl",
    "compute_logprobs",
    "compute_ratio",
    "create_shifted_targets",
    "masked_mean",
    "pg_ppo_clip",
]
