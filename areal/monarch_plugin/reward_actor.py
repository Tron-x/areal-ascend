"""RewardActor -- Monarch actor for distributed reward computation.

Phase 4: Replaces AReaL's AsyncRewardWrapper (ProcessPoolExecutor-based)
with a dedicated Monarch Actor, demonstrating:
  - Separate compute resource (CPU ProcMesh) for reward
  - Monarch RPC for inter-actor communication
  - Multi-actor coordination pattern

Architecture:
  RewardActor runs on its own ProcMesh (CPU-only, no NPU needed).
  MonarchRewardWrapper is a drop-in replacement for AsyncRewardWrapper:
  same async __call__ interface, but routes computation to RewardActor
  via Monarch RPC instead of a local ProcessPoolExecutor.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

from monarch.actor import endpoint

from areal.monarch_plugin.actor_base import MonarchActor
from areal.monarch_plugin.actor_spec import ResourceKind

logger = logging.getLogger(__name__)


class RewardActor(MonarchActor):
    """CPU-only Monarch Actor for reward computation."""

    resource = ResourceKind.CPU
    dependencies: list[str] = []
    """Monarch Actor that loads and executes a reward function.

    Lifecycle:
      __init__  -> empty state
      setup()   -> load reward function from import path
      compute_reward()  -> single reward computation
      compute_rewards_batch() -> batch reward computation
      shutdown() -> cleanup
    """

    def __init__(self):
        self._reward_fn = None
        self._fn_path: str | None = None
        self._call_count = 0
        self._total_time = 0.0

    @staticmethod
    def _patch_math_verify_timeout():
        """Replace math_verify's signal-based timeout with a no-op.

        math_verify uses ``signal.alarm()`` for timeout protection, which only
        works in the main thread of the main interpreter.  Monarch actors run
        endpoint handlers on asyncio/worker threads, so ``signal.alarm()``
        raises ``ValueError``.  We disable the timeout decorator entirely --
        the subprocess-level timeout in SandboxActor already guards against
        runaway code, and the reward parsing itself is fast enough.
        """

        def _noop_timeout(timeout_seconds=None):
            def decorator(func):
                return func

            return decorator

        try:
            import math_verify.grader
            import math_verify.metric
            import math_verify.parser
            import math_verify.utils

            math_verify.utils.timeout = _noop_timeout
            math_verify.parser.timeout = _noop_timeout
            math_verify.metric.timeout = _noop_timeout
            math_verify.grader.timeout = _noop_timeout
            logger.info(
                "[RewardActor] Patched math_verify timeout "
                "(disabled signal.alarm for non-main-thread compatibility)"
            )
        except ImportError:
            logger.debug("[RewardActor] math_verify not installed, skip patch")

    @endpoint
    def setup(self, reward_fn_path: str) -> dict:
        """Load a reward function by its import path.

        Parameters
        ----------
        reward_fn_path : str
            Dotted import path, e.g. ``"examples.math.reward.math_reward"``.
        """
        self._patch_math_verify_timeout()

        from areal.utils.dynamic_import import import_from_string

        self._reward_fn = import_from_string(reward_fn_path)
        self._fn_path = reward_fn_path
        logger.info(f"[RewardActor] Loaded reward function: {reward_fn_path}")
        return {"status": "ready", "reward_fn": reward_fn_path}

    @endpoint
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list,
        completion_ids: list,
        task_data: dict,
    ) -> float:
        """Compute reward for a single (prompt, completion) pair.

        Parameters
        ----------
        prompt : str
            The prompt text.
        completion : str
            The model-generated completion text.
        prompt_ids : list[int]
            Token IDs of the prompt.
        completion_ids : list[int]
            Token IDs of the completion.
        task_data : dict
            Extra keyword arguments from the dataset (e.g. solutions).

        Returns
        -------
        float
            Scalar reward value.
        """
        if self._reward_fn is None:
            raise RuntimeError(
                "RewardActor not set up. Call setup(reward_fn_path) first."
            )
        t0 = time.monotonic()
        try:
            reward = self._reward_fn(
                prompt, completion, prompt_ids, completion_ids, **task_data
            )
            reward = float(reward)
        except Exception:
            logger.error(
                f"[RewardActor] Error computing reward:\n{traceback.format_exc()}"
            )
            reward = 0.0

        elapsed = time.monotonic() - t0
        self._call_count += 1
        self._total_time += elapsed
        return reward

    @endpoint
    def compute_rewards_batch(self, items: list[dict]) -> list[float]:
        """Compute rewards for a batch of items.

        Parameters
        ----------
        items : list[dict]
            Each dict has keys: prompt, completion, prompt_ids,
            completion_ids, and optionally task_data.

        Returns
        -------
        list[float]
            Reward values, one per item.
        """
        if self._reward_fn is None:
            raise RuntimeError(
                "RewardActor not set up. Call setup(reward_fn_path) first."
            )
        rewards = []
        for item in items:
            try:
                r = self._reward_fn(
                    item["prompt"],
                    item["completion"],
                    item.get("prompt_ids", []),
                    item.get("completion_ids", []),
                    **item.get("task_data", {}),
                )
                rewards.append(float(r))
            except Exception:
                logger.error(
                    f"[RewardActor] Batch item error:\n{traceback.format_exc()}"
                )
                rewards.append(0.0)
        return rewards

    @endpoint
    def get_stats(self) -> dict:
        """Return reward computation statistics."""
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": avg,
            "reward_fn": self._fn_path,
        }

    @endpoint
    def shutdown(self) -> None:
        logger.info(
            f"[RewardActor] Shutting down. "
            f"Computed {self._call_count} rewards "
            f"(avg {self._total_time / max(1, self._call_count):.4f}s each)"
        )
        self._reward_fn = None


class MonarchRewardWrapper:
    """Drop-in replacement for ``AsyncRewardWrapper``.

    Same ``async __call__(prompt, completion, prompt_ids, completion_ids, **kwargs)``
    interface, but routes computation to a RewardActor via Monarch RPC
    instead of using a local ProcessPoolExecutor.

    Usage::

        wrapper = MonarchRewardWrapper(reward_actor, "my_module.reward_fn")
        reward = await wrapper(prompt, completion, prompt_ids, completion_ids, **data)
    """

    def __init__(self, reward_actor, reward_fn_path: str):
        self._actor = reward_actor
        self._fn_path = reward_fn_path
        self._setup_done = False

    async def _ensure_setup(self):
        if not self._setup_done:
            result = await self._actor.setup.call_one(self._fn_path)
            logger.info(f"[MonarchRewardWrapper] RewardActor setup: {result}")
            self._setup_done = True

    async def __call__(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list[int],
        completion_ids: list[int],
        **kwargs: Any,
    ) -> float:
        await self._ensure_setup()
        reward = await self._actor.compute_reward.call_one(
            prompt, completion, prompt_ids, completion_ids, kwargs
        )
        return reward
