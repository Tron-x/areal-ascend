"""RewardActor — framework-agnostic reward computation for RL training.

Depends ONLY on ``forge.core.protocols.RewardBackend``. Framework-specific
reward function loading lives in the adapter (e.g. ``AReaLRewardBackend``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

if TYPE_CHECKING:
    from forge.core.protocols import RewardBackend

logger = logging.getLogger(__name__)


class RewardActor(ForgeActor):
    """CPU-only Monarch Actor for reward computation.

    Accepts a ``RewardBackend`` instance that handles the actual reward
    function loading and computation. If no backend is provided, a default
    ``AReaLRewardBackend`` is created lazily.

    Deploy as a service for parallel reward evaluation::

        from forge.adapters.areal import AReaLRewardBackend
        reward = await RewardActor.options(
            num_replicas=2, procs=1
        ).as_actor(backend=AReaLRewardBackend())
        await reward.setup.call(reward_fn_path="areal.reward.gsm8k.gsm8k_reward_fn")
    """

    procs = 1
    with_gpus = False

    def __init__(self, backend: RewardBackend | None = None):
        self._backend = backend

    def _ensure_backend(self) -> RewardBackend:
        if self._backend is None:
            from forge.adapters.areal.reward_backend import AReaLRewardBackend

            self._backend = AReaLRewardBackend()
        return self._backend

    @endpoint
    def setup(self, reward_fn_path: str = "") -> dict:
        """Load the reward function via the backend."""
        return self._ensure_backend().setup(reward_fn_path)

    @endpoint
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float:
        """Compute reward for a single prompt-completion pair."""
        return self._ensure_backend().compute_reward(
            prompt=prompt,
            completion=completion,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            task_data=task_data,
        )

    @endpoint
    def compute_rewards_batch(self, items: list[dict]) -> list[float]:
        """Compute rewards for a batch of items."""
        return self._ensure_backend().compute_rewards_batch(items)

    @endpoint
    def get_stats(self) -> dict:
        return self._ensure_backend().get_stats()
