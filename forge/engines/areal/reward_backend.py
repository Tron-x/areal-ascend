"""AReaL reward backend — loads reward functions via import path.

Implements ``forge.core.protocols.RewardBackend``.
"""

from __future__ import annotations

import importlib
import logging
import time

logger = logging.getLogger(__name__)


class AReaLRewardBackend:
    """Reward backend that dynamically loads AReaL reward functions.

    Satisfies the ``RewardBackend`` protocol.
    """

    def __init__(self):
        self._reward_fn = None
        self._fn_path = ""
        self._call_count = 0
        self._total_time = 0.0

    def setup(self, reward_fn_path: str = "") -> dict:
        if not reward_fn_path:
            return {"reward_fn": "", "status": "not_configured"}
        self._fn_path = reward_fn_path
        module_path, fn_name = reward_fn_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        self._reward_fn = getattr(mod, fn_name)

        self._patch_math_verify_timeout()

        logger.info(f"Loaded reward function: {reward_fn_path}")
        return {"reward_fn": reward_fn_path, "status": "ready"}

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float:
        if self._reward_fn is None:
            raise RuntimeError("RewardBackend not set up. Call setup() first.")

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
            logger.warning(f"Reward computation failed: {e}, returning 0.0")
            reward = 0.0

        self._total_time += time.monotonic() - t0
        return reward

    def compute_rewards_batch(self, items: list[dict]) -> list[float]:
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

    def get_stats(self) -> dict:
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": avg,
            "reward_fn": self._fn_path,
        }

    @staticmethod
    def _patch_math_verify_timeout():
        """Patch math_verify timeout for non-main-thread execution."""
        try:
            from math_verify import verify as _mv

            if hasattr(_mv, "TIMEOUT"):
                _mv.TIMEOUT = 30
        except ImportError:
            pass
