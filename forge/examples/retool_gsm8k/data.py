"""Lightweight GSM8K data loader for Forge rollout.

Loads GSM8K from HuggingFace datasets (or local cache) and yields
formatted prompts suitable for the ReToolAgent.

No AReaL dependency -- uses HuggingFace datasets directly.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful math assistant. When you need to compute something, "
    "write Python code between <code> and </code> tags. The code will be "
    "executed and the result returned to you. When you have the final answer, "
    "write it as: Answer: \\boxed{your_answer}"
)


def load_gsm8k_prompts(
    split: str = "train",
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load GSM8K and format as Forge-compatible prompt dicts.

    Returns:
        List of dicts with keys: ``prompt``, ``answer``, ``messages``.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
    except Exception:
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "openai/gsm8k",
                "main",
                split=split,
                download_mode="reuse_cache_if_exists",
            )
        except Exception as e:
            logger.warning(f"Could not load GSM8K: {e}. Using dummy data.")
            return _dummy_data(max_samples or 10)

    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    prompts = []
    for item in ds:
        question = item["question"]
        raw_answer = item["answer"]
        final_answer = _extract_gsm8k_answer(raw_answer)

        prompts.append(
            {
                "prompt": question,
                "answer": final_answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            }
        )

    logger.info(f"Loaded {len(prompts)} GSM8K prompts (split={split})")
    return prompts


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K answer format.

    GSM8K answers end with ``#### <number>``.
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip().split("\n")[-1].strip()


def _dummy_data(n: int) -> list[dict]:
    """Fallback dummy data when GSM8K is unavailable."""
    problems = [
        ("What is 7 * 6?", "42"),
        ("What is 15 + 27?", "42"),
        ("If a store has 5 apples and gets 3 more, how many total?", "8"),
        ("What is 100 / 4?", "25"),
        ("A train travels 60 km/h for 2 hours. How far?", "120"),
    ]
    prompts = []
    for i in range(n):
        q, a = problems[i % len(problems)]
        prompts.append(
            {
                "prompt": q,
                "answer": a,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
            }
        )
    return prompts
