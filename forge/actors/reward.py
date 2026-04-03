"""RewardActor -- distributed reward computation for RL training.

Loads reward functions dynamically by import path. Deployed as a
service for load-balanced reward evaluation across replicas.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


class RewardActor(ForgeActor):
    """CPU-only Monarch Actor for reward computation.

    Deploy as a service for parallel reward evaluation::

        reward = await RewardActor.options(
            num_replicas=2, procs=1
        ).as_service()
        score = await reward.compute_reward.route(
            prompt="...", completion="...", task_data={...}
        )
    """

    procs = 1
    with_gpus = False

    def __init__(self):
        self._reward_fn = None
        self._fn_path = ""
        self._call_count = 0
        self._total_time = 0.0

    @endpoint
    def setup(self, reward_fn_path: str = "") -> dict:
        """Load the reward function from an import path.

        Args:
            reward_fn_path: Dotted path like ``areal.reward.gsm8k.gsm8k_reward_fn``.

        Returns:
            Dict with loaded function info.
        """
        if not reward_fn_path:
            return {"reward_fn": "", "status": "not_configured"}
        self._fn_path = reward_fn_path
        module_path, fn_name = reward_fn_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        self._reward_fn = getattr(mod, fn_name)

        self._patch_math_verify_timeout()

        logger.info(f"[RewardActor] Loaded reward function: {reward_fn_path}")
        return {"reward_fn": reward_fn_path, "status": "ready"}

    @endpoint
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float:
        """Compute reward for a single prompt-completion pair.

        Args:
            prompt: Input prompt text.
            completion: Model completion text.
            prompt_ids: Tokenized prompt (optional).
            completion_ids: Tokenized completion (optional).
            task_data: Additional task metadata.

        Returns:
            Reward score as float.
        """
        if self._reward_fn is None:
            raise RuntimeError("RewardActor not set up. Call setup() first.")

        t0 = time.monotonic()
        self._call_count += 1

        try:
            kwargs = {}
            if task_data:
                kwargs.update(task_data)
            reward = self._reward_fn(
                prompt, completion, prompt_ids, completion_ids, **kwargs
            )
            if isinstance(reward, dict):
                reward = reward.get("reward", 0.0)
            reward = float(reward)
        except Exception as e:
            logger.warning(
                f"[RewardActor] Reward computation failed: {e}, returning 0.0"
            )
            reward = 0.0

        self._total_time += time.monotonic() - t0
        return reward

    @endpoint
    def compute_rewards_batch(self, items: list[dict]) -> list[float]:
        """Compute rewards for a batch of items.

        Args:
            items: List of dicts with keys: prompt, completion,
                   prompt_ids, completion_ids, task_data.

        Returns:
            List of reward scores.
        """
        return [
            self.compute_reward(
                prompt=item.get("prompt", ""),
                completion=item.get("completion", ""),
                prompt_ids=item.get("prompt_ids"),
                completion_ids=item.get("completion_ids"),
                task_data=item.get("task_data"),
            )
            for item in items
        ]

    @endpoint
    def get_stats(self) -> dict:
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": avg,
            "reward_fn": self._fn_path,
        }

    def _patch_math_verify_timeout(self):
        """Patch math_verify timeout for non-main-thread execution."""
        try:
            from math_verify import verify as _mv

            if hasattr(_mv, "TIMEOUT"):
                _mv.TIMEOUT = 30
        except ImportError:
            pass


class MonarchRewardWrapper:
    """Async reward wrapper that calls RewardActor via Monarch RPC.

    Drop-in replacement for ``AsyncRewardWrapper`` used by rollout workflows.
    """

    def __init__(self, reward_actor, reward_fn_path: str = ""):
        self._reward = reward_actor
        self._reward_fn_path = reward_fn_path

    async def __call__(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list[int] | None = None,
        completion_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> float:
        task_data = {
            k: v for k, v in kwargs.items() if k != "prompt" and k != "completion"
        }

        if hasattr(self._reward, "compute_reward"):
            if hasattr(self._reward.compute_reward, "route"):
                return await self._reward.compute_reward.route(
                    prompt=prompt,
                    completion=completion,
                    prompt_ids=list(prompt_ids) if prompt_ids else None,
                    completion_ids=list(completion_ids) if completion_ids else None,
                    task_data=task_data,
                )
            else:
                return await self._reward.compute_reward.call_one(
                    prompt=prompt,
                    completion=completion,
                    prompt_ids=list(prompt_ids) if prompt_ids else None,
                    completion_ids=list(completion_ids) if completion_ids else None,
                    task_data=task_data,
                )
        raise RuntimeError("RewardActor has no compute_reward endpoint")
