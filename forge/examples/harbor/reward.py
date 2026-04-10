"""Harbor reward functions wrapped for the Forge RewardFn protocol.

Bridges rllm's ``RewardFunction`` protocol (which takes ``(task_info, action)``)
to Forge's ``RewardFn`` protocol (which takes ``(prompt, response, target, **kwargs)``).

Provides ready-to-use reward functions for math and code tasks, plus a
generic wrapper for any rllm ``RewardFunction``.

Usage::

    from forge.examples.harbor.reward import harbor_math_reward

    # As a Forge RewardFn:
    score = harbor_math_reward(prompt="...", response="...", target="42")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_RLLM_AVAILABLE = None


def _ensure_rllm():
    """Lazy-check that rllm is importable; add to sys.path if needed."""
    global _RLLM_AVAILABLE
    if _RLLM_AVAILABLE is not None:
        return _RLLM_AVAILABLE

    try:
        import rllm.rewards  # noqa: F401

        _RLLM_AVAILABLE = True
    except ImportError:
        harbor_root = "/root/harbor/harbor-verl-train"
        if harbor_root not in sys.path:
            sys.path.insert(0, harbor_root)
        try:
            import rllm.rewards  # noqa: F401

            _RLLM_AVAILABLE = True
        except ImportError:
            logger.warning(
                "rllm package not found. Harbor reward functions will return 0.0. "
                "Install with: pip install -e /root/harbor/harbor-verl-train"
            )
            _RLLM_AVAILABLE = False
    return _RLLM_AVAILABLE


def wrap_rllm_reward(rllm_reward_fn: Any) -> Any:
    """Wrap an rllm ``RewardFunction`` as a Forge-compatible reward callable.

    Args:
        rllm_reward_fn: A callable with signature ``(task_info, action) -> RewardOutput``.

    Returns:
        A callable with Forge's ``RewardFn`` signature:
        ``(prompt, response, target, **kwargs) -> float``.
    """

    def forge_reward(
        prompt: str,
        response: str,
        target: Any = None,
        **kwargs: Any,
    ) -> float:
        thought_end = "</think>"
        if thought_end in response:
            action_text = response.split(thought_end, 1)[1].strip()
        else:
            action_text = response

        task_info = {
            "problem": prompt,
            "ground_truth": target or kwargs.get("answer", ""),
            "data_source": kwargs.get("data_source", ""),
            **{k: v for k, v in kwargs.items() if k not in ("answer", "data_source")},
        }

        try:
            result = rllm_reward_fn(task_info, action_text)
            if hasattr(result, "reward"):
                return float(result.reward)
            return float(result)
        except Exception as e:
            logger.warning("Harbor reward computation failed: %s", e)
            return 0.0

    forge_reward.__name__ = getattr(rllm_reward_fn, "__name__", "harbor_reward")
    forge_reward.__qualname__ = f"wrap_rllm_reward({forge_reward.__name__})"
    return forge_reward


def harbor_math_reward(
    prompt: str,
    response: str,
    target: Any = None,
    **kwargs: Any,
) -> float:
    """Forge-compatible math reward using rllm's ``math_reward_fn``.

    Extracts the answer from ``\\boxed{...}`` in the response and compares
    against the ground truth using symbolic and string matching.

    Args:
        prompt: The math problem text.
        response: The model's full response (may include ``<think>...</think>``).
        target: The ground-truth answer string.
        **kwargs: Extra fields forwarded to the rllm reward function.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    if not _ensure_rllm():
        return _fallback_math_reward(response, target)

    from rllm.rewards.reward_fn import math_reward_fn

    return wrap_rllm_reward(math_reward_fn)(
        prompt=prompt, response=response, target=target, **kwargs
    )


def harbor_code_reward(
    prompt: str,
    response: str,
    target: Any = None,
    **kwargs: Any,
) -> float:
    """Forge-compatible code reward using rllm's ``code_reward_fn``."""
    if not _ensure_rllm():
        return 0.0

    from rllm.rewards.reward_fn import code_reward_fn

    return wrap_rllm_reward(code_reward_fn)(
        prompt=prompt, response=response, target=target, **kwargs
    )


def _fallback_math_reward(response: str, target: Any) -> float:
    """Simple fallback when rllm is not installed."""
    import re

    if target is None:
        return 0.0

    match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", response)
    if not match:
        return 0.0

    predicted = match.group(1).strip().lower().replace(",", "").replace("$", "")
    expected = str(target).strip().lower().replace(",", "").replace("$", "")
    return 1.0 if predicted == expected else 0.0
