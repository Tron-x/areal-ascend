"""FSDP batch adapter — converts Episode objects to FSDP TrainEngine batch format.

Implements ``forge.core.protocols.BatchAdapter`` for the FSDP2 training
engine.  The FSDP engine's ``train_step`` expects:

    input_ids          -- [B, seq_len] token IDs (prompt + completion)
    attention_mask     -- [B, seq_len] 1=real, 0=pad
    loss_mask          -- [B, seq_len] 1=trainable tokens, 0=masked
    advantages         -- [B, seq_len] per-token advantages (broadcast from scalar)
    generator_logprobs -- [B, seq_len] old-policy log probs
    ref_logprobs       -- [B, seq_len] reference log probs (optional)

This adapter handles padding, advantage broadcasting, and optional
truncation.
"""

from __future__ import annotations

from typing import Any

from forge.core.types import Episode


class FSDPBatchAdapter:
    """Convert ``Episode`` objects to FSDP TrainEngine batch format.

    Args:
        pad_token_id: Token ID for padding.
        max_seq_len: Hard cap on sequence length (truncates from right).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_len: int | None = None,
    ):
        self._pad_id = pad_token_id
        self._max_seq_len = max_seq_len

    def adapt(self, episodes: list[Episode | dict]) -> dict[str, Any]:
        if not episodes:
            return {}

        normalized = [self._normalize(ep) for ep in episodes]

        max_len = max(len(ep["token_ids"]) for ep in normalized)
        if self._max_seq_len is not None:
            max_len = min(max_len, self._max_seq_len)

        batch: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": [],
            "advantages": [],
            "generator_logprobs": [],
        }
        has_ref = any(ep.get("ref_logprobs") for ep in normalized)
        if has_ref:
            batch["ref_logprobs"] = []

        for ep in normalized:
            seq_len = len(ep["token_ids"])
            trunc = min(seq_len, max_len)
            pad = max_len - trunc

            batch["input_ids"].append(
                ep["token_ids"][:trunc] + [self._pad_id] * pad
            )
            batch["attention_mask"].append([1] * trunc + [0] * pad)
            batch["loss_mask"].append(
                ep["loss_mask"][:trunc] + [0] * pad
            )

            adv = ep["advantage"]
            adv_seq = [adv] * trunc + [0.0] * pad
            batch["advantages"].append(adv_seq)

            batch["generator_logprobs"].append(
                ep["logprobs"][:trunc] + [0.0] * pad
            )

            if has_ref:
                ref = ep.get("ref_logprobs", [0.0] * seq_len)
                batch["ref_logprobs"].append(ref[:trunc] + [0.0] * pad)

        return batch

    def required_fields(self) -> list[str]:
        return ["token_ids", "loss_mask", "generator_logprobs", "reward"]

    def _normalize(self, ep: Episode | dict) -> dict[str, Any]:
        if isinstance(ep, Episode):
            token_ids = ep.token_ids
            logprobs = ep.generator_logprobs
            loss_mask = ep.loss_mask
            advantage = getattr(ep, "advantage", None) or ep.reward
            ref_logprobs = getattr(ep, "ref_logprobs", None)
        elif isinstance(ep, dict):
            token_ids = ep.get("token_ids", ep.get("input_ids", []))
            logprobs = ep.get("generator_logprobs", ep.get("logprobs", []))
            loss_mask = ep.get("loss_mask", [])
            advantage = ep.get("advantage", ep.get("reward", 0.0))
            ref_logprobs = ep.get("ref_logprobs")
            if isinstance(advantage, list):
                advantage = advantage[0] if advantage else 0.0
        else:
            raise TypeError(f"Expected Episode or dict, got {type(ep)}")

        seq_len = len(token_ids)
        if not logprobs:
            logprobs = [0.0] * seq_len
        if not loss_mask:
            loss_mask = [1] * seq_len

        return {
            "token_ids": token_ids,
            "logprobs": logprobs,
            "loss_mask": loss_mask,
            "advantage": float(advantage),
            "ref_logprobs": ref_logprobs,
        }
