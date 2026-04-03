"""Reward function protocol for Forge.

Reward functions can be:
1. Simple callables matching the ``RewardFn`` protocol.
2. Dynamic imports from dotted paths (e.g. ``"areal.reward.gsm8k.gsm8k_reward_fn"``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RewardFn(Protocol):
    """Protocol for reward computation.

    Implementations receive a prompt, completion, their token IDs,
    and optional task-specific keyword arguments (e.g. ground-truth labels).
    """

    def __call__(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list[int],
        completion_ids: list[int],
        **task_data: Any,
    ) -> float:
        """Compute a scalar reward for a (prompt, completion) pair.

        Parameters
        ----------
        prompt
            The input prompt text.
        completion
            The model-generated completion text.
        prompt_ids
            Token IDs for the prompt.
        completion_ids
            Token IDs for the completion.
        **task_data
            Additional fields from the dataset (e.g. ``solutions``).

        Returns
        -------
        float
            Scalar reward value.
        """
        ...
