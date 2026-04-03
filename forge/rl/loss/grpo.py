"""DR-GRPO loss: "Done Right" Group Relative Policy Optimization.

Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
https://arxiv.org/abs/2503.20783

Per-token: ``L_t = max(-r*A, -clip(r, 1-eps, 1+eps)*A) + beta*KL``
Aggregated: ``L = sum(L_t * mask) / (B * MAX_LEN)``

Key design choices (DR-GRPO vs vanilla GRPO):
1. Fixed-horizon aggregation removes length bias.
2. No std normalization in advantages removes difficulty bias.
3. Asymmetric clip (clip_high > clip_low) encourages exploration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    pg_ppo_clip,
)
from forge.rl.loss.types import AggType, LossOutput


@dataclass
class GRPOLoss:
    """DR-GRPO loss function.

    Args:
        clip_low: Lower clip bound (default 0.2).
        clip_high: Upper clip bound (default 0.28).
        beta: KL penalty coefficient (default 0.1). Set 0 to disable.
        agg_type: Aggregation strategy (default ``"fixed_horizon"``).
    """

    clip_low: float = 0.2
    clip_high: float = 0.28
    beta: float = 0.1
    agg_type: AggType = "fixed_horizon"

    def __call__(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        advantages: torch.Tensor,
        generator_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        ref_logprobs: torch.Tensor | None = None,
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

        kl_m = []
        if self.beta > 0:
            if ref_logprobs is None:
                raise ValueError("ref_logprobs required when beta > 0")
            kl, kl_m = compute_kl(logprobs, ref_logprobs, loss_mask)
            pg_loss = pg_loss + self.beta * kl

        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)

        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + kl_m + agg_m)
