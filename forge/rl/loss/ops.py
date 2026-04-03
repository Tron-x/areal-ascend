"""Composable RL loss primitives.

Each function returns ``(result_tensor, list[Metric])`` so that metrics
accumulate alongside the computation without side effects.

Adapted from TorchForge (Meta), MIT-licensed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from forge.rl.loss.types import (
    CROSS_ENTROPY_IGNORE_IDX,
    AggType,
    KLType,
    Metric,
    RatioType,
)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked mean: ``sum(values * mask) / divisor``.

    Args:
        values: Per-token values ``(B, S)``.
        mask: Valid token mask ``(B, S)``.
        loss_scale: If provided, use as divisor instead of ``mask.sum()``.
    """
    masked_sum = (values * mask).sum()
    if loss_scale is not None:
        divisor = loss_scale.clamp(min=1.0)
    else:
        divisor = mask.sum().clamp(min=1.0)
    return masked_sum / divisor


def create_shifted_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.Tensor:
    """Create next-token prediction targets via ``torch.roll``.

    ``targets[i] = input_ids[i+1]``, last position → ``ignore_index``.
    """
    targets = torch.roll(input_ids, shifts=-1, dims=-1)
    if input_ids.dim() == 1:
        targets[-1] = ignore_index
    else:
        targets[:, -1] = ignore_index

    if loss_mask is not None:
        targets = torch.where(
            loss_mask.bool(), targets, torch.full_like(targets, ignore_index)
        )
    return targets


def compute_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = CROSS_ENTROPY_IGNORE_IDX,
) -> tuple[torch.Tensor, list[Metric]]:
    """Compute per-token log-probs via negative cross-entropy.

    Casts to fp32 before temperature division for numerical precision.
    """
    logits_fp32 = logits.float() / temperature
    B, S, V = logits_fp32.shape
    logprobs = -F.cross_entropy(
        logits_fp32.view(-1, V),
        target_ids.view(-1).long(),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, S)
    return logprobs, []


def compute_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, list[Metric]]:
    """Numerically stable per-token entropy: ``H = logsumexp - sum(p * logits)``."""
    logits_fp32 = logits.float()
    probs = F.softmax(logits_fp32, dim=-1)
    entropy = torch.logsumexp(logits_fp32, dim=-1) - (probs * logits_fp32).sum(dim=-1)

    with torch.no_grad():
        metrics = [
            Metric("loss/entropy/mean", masked_mean(entropy, mask)),
        ]
    return entropy, metrics


def compute_ratio(
    logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    mask: torch.Tensor,
    ratio_type: RatioType = "token",
) -> tuple[torch.Tensor, torch.Tensor, list[Metric]]:
    """Importance sampling ratio ``r = π_θ / π_old``.

    Args:
        ratio_type: ``"token"`` for per-token, ``"sequence"`` for
            per-sequence (GSPO-style reparameterization).
    """
    if ratio_type == "token":
        log_ratio = logprobs - generator_logprobs.detach()
        ratio = torch.exp(log_ratio)
    elif ratio_type == "sequence":
        token_log_ratio = logprobs - generator_logprobs.detach()
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (token_log_ratio * mask).sum(dim=-1) / seq_lengths
        log_ratio = logprobs - logprobs.detach() + seq_log_ratio.detach().unsqueeze(-1)
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(f"Unknown ratio_type: {ratio_type}")

    with torch.no_grad():
        metrics = [
            Metric("loss/ratio/mean", masked_mean(ratio, mask)),
            Metric("loss/kl_policy/mean", masked_mean(-log_ratio, mask)),
        ]
    return ratio, log_ratio, metrics


def compute_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    kl_type: KLType = "k3",
) -> tuple[torch.Tensor, list[Metric]]:
    """Per-token KL divergence (Schulman's k1/k2/k3 estimators).

    k3 (default): unbiased KL with low variance, ``r_ref - log(r_ref) - 1``.
    """
    log_ratio = policy_logprobs - ref_logprobs.detach()

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio.square()
    elif kl_type == "k3":
        neg_log_ratio = torch.clamp(-log_ratio, min=-10.0, max=10.0)
        ratio = torch.exp(neg_log_ratio)
        kl = ratio - neg_log_ratio - 1
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    with torch.no_grad():
        metrics = [Metric("loss/kl_ref/mean", masked_mean(kl, mask))]
    return kl, metrics


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, list[Metric]]:
    """PPO clipped surrogate: ``L = max(-r*A, -clip(r)*A)``.

    Supports asymmetric clip (``clip_high > clip_low``).
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        clipped_high = (ratio > 1 + clip_high) & mask.bool()
        clipped_low = (ratio < 1 - clip_low) & mask.bool()
        pos_adv = advantages > 0
        neg_adv = advantages < 0
        metrics = [
            Metric("loss/clip/clipped_ratio/mean", masked_mean(clipped_ratio, mask)),
            Metric(
                "loss/clip/high_fraction",
                masked_mean((clipped_high & pos_adv).float(), mask),
            ),
            Metric(
                "loss/clip/low_fraction",
                masked_mean((clipped_low & neg_adv).float(), mask),
            ),
        ]
    return pg_loss, metrics


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    agg_type: AggType = "token_mean",
    loss_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[Metric]]:
    """Aggregate per-token loss to scalar.

    - ``token_mean``: ``sum(L*mask) / sum(mask)``
    - ``fixed_horizon``: ``sum(L*mask) / (B*S)`` (DR-GRPO, no length bias)
    - ``sequence_mean``: per-sequence mean then batch mean
    """
    if agg_type == "token_mean":
        loss = masked_mean(per_token_loss, mask, loss_scale)
    elif agg_type == "fixed_horizon":
        loss = (per_token_loss * mask).sum() / max(mask.numel(), 1)
    elif agg_type == "sequence_mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_means = (per_token_loss * mask).sum(dim=-1) / seq_lengths
        loss = seq_means.sum() / max(seq_means.numel(), 1)
    else:
        raise ValueError(f"Unknown agg_type: {agg_type}")

    with torch.no_grad():
        metrics = [Metric("loss/aggregate/active_fraction", mask.mean())]
    return loss, metrics
