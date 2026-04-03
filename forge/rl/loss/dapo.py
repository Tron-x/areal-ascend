"""DAPO loss: Decoupled clip + Dynamic sAmpling Policy Optimization.

Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System
at Scale" (2025). https://arxiv.org/abs/2503.14476

Key differences from GRPO:
- Asymmetric clip (clip_high > clip_low) for more exploration.
- Dual-clip: caps penalty on negative advantages (``min(L_PPO, -c*A)``).
- Token-level aggregation (divides by total trainable tokens).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_logprobs,
    compute_ratio,
    masked_mean,
    pg_ppo_clip,
)
from forge.rl.loss.types import AggType, LossOutput, Metric


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    c: float = 3.0,
) -> tuple[torch.Tensor, list[Metric]]:
    """DAPO dual-clip: ``L = min(L_PPO, -c*A)`` when ``A < 0``.

    Prevents over-penalization of "wrong" tokens that are actually
    productive exploration.
    """
    dual_clip_bound = -c * advantages
    loss = torch.where(
        advantages < 0,
        torch.minimum(pg_loss, dual_clip_bound),
        pg_loss,
    )

    with torch.no_grad():
        neg_mask = (advantages < 0) & mask.bool()
        was_dual_clipped = (pg_loss > dual_clip_bound) & neg_mask
        metrics = [
            Metric(
                "loss/dual_clip/clip_fraction",
                masked_mean(was_dual_clipped.float(), mask),
            ),
        ]
    return loss, metrics


@dataclass
class DAPOLoss:
    """DAPO loss function.

    Args:
        clip_low: Lower clip bound (default 0.2).
        clip_high: Upper clip bound (default 0.28).
        dual_clip_c: Dual-clip constant (default 3.0).
        agg_type: Aggregation strategy (default ``"token_mean"``).
    """

    clip_low: float = 0.2
    clip_high: float = 0.28
    dual_clip_c: float = 3.0
    agg_type: AggType = "token_mean"

    def __call__(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        advantages: torch.Tensor,
        generator_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_scale: torch.Tensor | None = None,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, generator_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )
        pg_loss, dual_m = pg_dual_clip(pg_loss, advantages, loss_mask, self.dual_clip_c)
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + dual_m + agg_m)
