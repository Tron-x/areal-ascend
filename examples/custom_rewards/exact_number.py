"""A stricter variant of gsm8k reward: require EXACT last-number match.

Algorithm engineer's use case: gsm8k's default ``math_verify`` reward
is lenient (allows ``16.0``, ``16``, ``+16`` all to count).  We want a
baseline that scores 1.0 only when the last number in the completion
is a character-for-character match with the ground truth answer.
"""

from __future__ import annotations

import re

from forge.reward import register_reward


def _last_number(text: str) -> str | None:
    """Return the last numeric token in ``text`` (or None)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


@register_reward("exact_number")
def exact_number_reward(
    prompt, completions, prompt_ids=None, completion_ids=None, answer=None, **kwargs
) -> float:
    if answer is None:
        return 0.0
    pred = _last_number(str(completions))
    gold = _last_number(str(answer)) or str(answer).strip()
    if pred is None or gold is None:
        return 0.0
    return 1.0 if pred == gold else 0.0
