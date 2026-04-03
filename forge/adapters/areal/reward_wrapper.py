"""Monarch reward wrapper for AReaL workflows.

``MonarchRewardWrapper`` is an async callable that routes reward computation
through a ``RewardActor`` via Monarch RPC. It serves as a drop-in replacement
for AReaL's ``AsyncRewardWrapper`` used by rollout workflows.
"""

from __future__ import annotations

from typing import Any


class MonarchRewardWrapper:
    """Async reward wrapper that calls RewardActor via Monarch RPC."""

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
