"""AReaL batch adapter -- converts Episode objects to AReaL's tensor dict.

Implements ``forge.core.protocols.BatchAdapter`` for the AReaL training
backend (PPOTrainer + FSDPEngine).

AReaL's PPOTrainer expects a dict with these tensor keys (each shaped
``[batch_size, seq_len]``)::

    input_ids        -- int64, token IDs (prompt + completion)
    attention_mask   -- int64, 1 for real tokens, 0 for padding
    loss_mask        -- int64, 1 for tokens to train on, 0 otherwise
    logprobs         -- float32, per-token log-probabilities from generator
    versions         -- int64, per-token policy version tags
    rewards          -- float32, [batch_size] scalar rewards

This adapter pads variable-length episodes to the same seq_len within
a batch and constructs the above layout from ``Episode`` fields.

When switching to a different training engine (TorchTitan, Slime, ...),
you only need to write a new ``BatchAdapter`` -- no actor changes required.
"""

from __future__ import annotations

from typing import Any

from forge.core.types import Episode


class AReaLBatchAdapter:
    """Convert ``Episode`` objects to AReaL PPOTrainer batch format.

    Satisfies the ``BatchAdapter`` protocol.

    Args:
        pad_token_id: Token ID used for padding shorter sequences.
            Defaults to 0 (common for most tokenizers).
        max_seq_len: Optional hard cap on sequence length. Episodes
            longer than this are truncated from the right.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_len: int | None = None,
    ):
        self._pad_id = pad_token_id
        self._max_seq_len = max_seq_len

    def adapt(self, episodes: list[Episode | dict]) -> dict[str, Any]:
        """Convert episodes to AReaL's expected batch dict.

        Accepts both ``Episode`` objects and legacy dicts (for backward
        compatibility with existing rollout code).

        Returns:
            Dict with keys ``input_ids``, ``attention_mask``, ``loss_mask``,
            ``logprobs``, ``versions``, ``rewards`` -- each as a list of
            lists (batch_size × seq_len) or list of floats (rewards).
        """
        if not episodes:
            return {}

        normalized = [self._normalize(ep) for ep in episodes]

        max_len = max(len(ep["token_ids"]) for ep in normalized)
        if self._max_seq_len is not None:
            max_len = min(max_len, self._max_seq_len)

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_loss_mask: list[list[int]] = []
        batch_logprobs: list[list[float]] = []
        batch_versions: list[list[int]] = []
        batch_rewards: list[float] = []

        for ep in normalized:
            seq_len = len(ep["token_ids"])
            trunc = min(seq_len, max_len)
            pad = max_len - trunc

            batch_input_ids.append(ep["token_ids"][:trunc] + [self._pad_id] * pad)
            batch_attention_mask.append([1] * trunc + [0] * pad)
            batch_loss_mask.append(ep["loss_mask"][:trunc] + [0] * pad)
            batch_logprobs.append(ep["logprobs"][:trunc] + [0.0] * pad)
            batch_versions.append(ep["versions"][:trunc] + [-1] * pad)
            batch_rewards.append(ep["reward"])

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "logprobs": batch_logprobs,
            "versions": batch_versions,
            "rewards": batch_rewards,
        }

    def required_fields(self) -> list[str]:
        return ["token_ids", "loss_mask", "generator_logprobs", "versions", "reward"]

    def _normalize(self, ep: Episode | dict) -> dict[str, Any]:
        """Extract the fields we need, handling both Episode and dict."""
        if isinstance(ep, Episode):
            token_ids = ep.token_ids
            logprobs = ep.generator_logprobs
            loss_mask = ep.loss_mask
            versions = ep.versions
            reward = ep.reward
        elif isinstance(ep, dict):
            token_ids = ep.get("token_ids", ep.get("input_ids", []))
            logprobs = ep.get("generator_logprobs", ep.get("logprobs", []))
            loss_mask = ep.get("loss_mask", [])
            versions = ep.get("versions", [])
            reward = ep.get("reward", ep.get("rewards", 0.0))
            if isinstance(reward, list):
                reward = reward[0] if reward else 0.0
        else:
            raise TypeError(f"Expected Episode or dict, got {type(ep)}")

        seq_len = len(token_ids)
        if not logprobs:
            logprobs = [0.0] * seq_len
        if not loss_mask:
            loss_mask = [1] * seq_len
        if not versions:
            versions = [-1] * seq_len

        return {
            "token_ids": token_ids,
            "logprobs": logprobs,
            "loss_mask": loss_mask,
            "versions": versions,
            "reward": float(reward),
        }
