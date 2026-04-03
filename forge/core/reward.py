"""Reward computation helpers for Forge.

Pure Python -- no framework dependency. Provides dynamic import
and batch computation utilities.
"""

from __future__ import annotations

import importlib
import logging
import time
import traceback
from typing import Any

from forge.api.reward import RewardFn

logger = logging.getLogger("forge.reward")


def load_reward_fn(import_path: str) -> RewardFn:
    """Load a reward function from a dotted import path.

    Parameters
    ----------
    import_path
        e.g. ``"areal.reward.gsm8k.gsm8k_reward_fn"``

    Returns
    -------
    RewardFn
        The loaded callable.
    """
    parts = import_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid reward_fn path '{import_path}'. "
            "Expected 'module.path.function_name'."
        )
    module_path, fn_name = parts
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"{import_path} is not callable")
    return fn


class RewardComputer:
    """Wraps a RewardFn with error handling and statistics tracking."""

    def __init__(self, reward_fn: RewardFn | str) -> None:
        if isinstance(reward_fn, str):
            self._fn = load_reward_fn(reward_fn)
            self._fn_path = reward_fn
        else:
            self._fn = reward_fn
            self._fn_path = (
                getattr(reward_fn, "__module__", "")
                + "."
                + getattr(reward_fn, "__name__", "unknown")
            )

        self._call_count = 0
        self._total_time = 0.0

    def compute(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list[int] | None = None,
        completion_ids: list[int] | None = None,
        **task_data: Any,
    ) -> float:
        """Compute reward for a single (prompt, completion) pair."""
        t0 = time.monotonic()
        self._call_count += 1
        try:
            reward = self._fn(
                prompt,
                completion,
                prompt_ids or [],
                completion_ids or [],
                **task_data,
            )
            return float(reward)
        except Exception:
            logger.error("Reward computation error:\n%s", traceback.format_exc())
            return 0.0
        finally:
            self._total_time += time.monotonic() - t0

    def compute_batch(self, items: list[dict]) -> list[float]:
        """Compute rewards for a batch of items.

        Each dict should have keys: prompt, completion, and optionally
        prompt_ids, completion_ids, task_data.
        """
        return [
            self.compute(
                item["prompt"],
                item["completion"],
                item.get("prompt_ids"),
                item.get("completion_ids"),
                **item.get("task_data", {}),
            )
            for item in items
        ]

    def stats(self) -> dict:
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": avg,
            "reward_fn": self._fn_path,
        }
