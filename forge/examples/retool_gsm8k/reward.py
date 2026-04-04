"""GSM8K math reward function.

Checks if the model's answer matches the ground truth.
Uses math_verify if available, falls back to string matching.
"""

from __future__ import annotations

import re


def gsm8k_reward_fn(
    prompt: str,
    response: str,
    target: str | None = None,
    **kwargs,
) -> float:
    """Compute binary reward for GSM8K math problems.

    Extracts the answer from ``\\boxed{...}`` and compares to target.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    if target is None:
        answer = kwargs.get("answer", "")
    else:
        answer = str(target)

    predicted = extract_boxed_answer(response)
    if predicted is None:
        return 0.0

    try:
        from areal.reward import get_math_verify_worker

        worker = get_math_verify_worker()
        return float(worker.verify(predicted, answer))
    except (ImportError, Exception):
        return 1.0 if normalize_answer(predicted) == normalize_answer(answer) else 0.0


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from ``\\boxed{...}`` pattern."""
    pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.strip()
    return s
