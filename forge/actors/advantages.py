"""ComputeAdvantages actor for RL advantage estimation.

Standalone actor that computes advantages from reward signals,
decoupled from the trainer to enable async pipeline operation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


class ComputeAdvantages(ForgeActor):
    """CPU actor for advantage computation.

    Computes advantages using GRPO-style group normalization or
    GAE, depending on configuration.

    Deploy as a single actor::

        compute_adv = await ComputeAdvantages.options(procs=1).as_actor()
        adv_batch = await compute_adv.compute.call_one(batch_data)
    """

    procs = 1
    with_gpus = False

    def __init__(self, gamma: float = 1.0, lam: float = 1.0, use_grpo: bool = True):
        self._gamma = gamma
        self._lam = lam
        self._use_grpo = use_grpo

    @endpoint
    def compute(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute advantages for a rollout batch.

        For GRPO: group-normalize rewards to compute advantages.
        For GAE: use generalized advantage estimation with values.

        Args:
            batch: Dict containing ``rewards`` (and optionally ``values``).

        Returns:
            Dict with ``advantages`` and ``returns`` added.
        """
        rewards = batch.get("rewards")
        if rewards is None:
            raise ValueError("Batch must contain 'rewards' key")

        if isinstance(rewards, list):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        if self._use_grpo:
            return self._grpo_advantages(batch, rewards)
        else:
            return self._gae_advantages(batch, rewards)

    def _grpo_advantages(self, batch: dict, rewards: torch.Tensor) -> dict[str, Any]:
        """GRPO-style: normalize rewards within group."""
        if rewards.numel() > 1:
            mean = rewards.mean()
            std = rewards.std().clamp(min=1e-8)
            advantages = (rewards - mean) / std
        else:
            advantages = rewards

        result = dict(batch)
        result["advantages"] = advantages
        result["returns"] = rewards
        return result

    def _gae_advantages(self, batch: dict, rewards: torch.Tensor) -> dict[str, Any]:
        """Generalized Advantage Estimation."""
        values = batch.get("values")
        if values is None:
            return self._grpo_advantages(batch, rewards)

        if isinstance(values, list):
            values = torch.tensor(values, dtype=torch.float32)

        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self._gamma * next_value - values[t]
            gae = delta + self._gamma * self._lam * gae
            advantages[t] = gae

        returns = advantages + values[:T]

        result = dict(batch)
        result["advantages"] = advantages
        result["returns"] = returns
        return result
