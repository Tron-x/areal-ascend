"""Data loaders for Harbor tasks, formatted for Forge rollout.

Supports loading from:
- HuggingFace datasets (GSM8K, MATH, etc.)
- Parquet files (Harbor preprocessed datasets)
- JSONL files (custom task lists)

Each loader returns a list of task dicts compatible with both the
``HarborAgentLogic`` and the rllm ``Workflow.run(task, uid)`` interface.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Show your reasoning inside <think>...</think> tags. "
    "Put your final answer in \\boxed{your_answer}."
)

MATH_TOOL_SYSTEM_PROMPT = (
    "You are a helpful math assistant. When you need to compute something, "
    "write Python code between ```python and ``` tags. The code will be "
    "executed and the result returned to you. When you have the final answer, "
    "write it as \\boxed{your_answer}."
)


def load_gsm8k(
    split: str = "train",
    max_samples: int | None = None,
    use_tools: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load GSM8K and format as Harbor-compatible task dicts.

    Args:
        split: Dataset split (``"train"`` or ``"test"``).
        max_samples: Limit number of samples.
        use_tools: If True, use the tool-use system prompt.
        seed: Random seed for shuffling.

    Returns:
        List of task dicts with keys: ``question``, ``ground_truth``,
        ``messages``, ``data_source``.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
    except Exception as e:
        logger.warning("Could not load GSM8K from HuggingFace: %s", e)
        return _dummy_math_tasks(max_samples or 5, use_tools)

    ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    sys_prompt = MATH_TOOL_SYSTEM_PROMPT if use_tools else MATH_SYSTEM_PROMPT
    tasks = []
    for item in ds:
        answer = _extract_gsm8k_answer(item["answer"])
        tasks.append({
            "question": item["question"],
            "ground_truth": answer,
            "data_source": "gsm8k",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": item["question"]},
            ],
        })

    logger.info("Loaded %d GSM8K tasks (split=%s, tools=%s)", len(tasks), split, use_tools)
    return tasks


def load_math(
    split: str = "train",
    max_samples: int | None = None,
    use_tools: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load MATH dataset and format as Harbor-compatible task dicts.

    Args:
        split: Dataset split.
        max_samples: Limit number of samples.
        use_tools: If True, use the tool-use system prompt.
        seed: Random seed for shuffling.

    Returns:
        List of task dicts.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("hendrycks/competition_math", split=split)
    except Exception as e:
        logger.warning("Could not load MATH: %s", e)
        return _dummy_math_tasks(max_samples or 5, use_tools)

    ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    sys_prompt = MATH_TOOL_SYSTEM_PROMPT if use_tools else MATH_SYSTEM_PROMPT
    tasks = []
    for item in ds:
        tasks.append({
            "question": item["problem"],
            "ground_truth": item["solution"],
            "data_source": "math",
            "problem_type": item.get("type", ""),
            "level": item.get("level", ""),
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": item["problem"]},
            ],
        })

    logger.info("Loaded %d MATH tasks (split=%s)", len(tasks), split)
    return tasks


def load_parquet(
    path: str,
    max_samples: int | None = None,
    question_key: str = "question",
    answer_key: str = "ground_truth",
) -> list[dict[str, Any]]:
    """Load tasks from a Harbor-format Parquet file.

    Args:
        path: Path to the ``.parquet`` file.
        question_key: Column name for the question/prompt.
        answer_key: Column name for the ground truth answer.
        max_samples: Limit number of samples.

    Returns:
        List of task dicts.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet loading: pip install pandas pyarrow")

    df = pd.read_parquet(path)
    if max_samples:
        df = df.head(max_samples)

    tasks = []
    for _, row in df.iterrows():
        task = row.to_dict()
        if question_key in task and "question" not in task:
            task["question"] = task[question_key]
        if answer_key in task and "ground_truth" not in task:
            task["ground_truth"] = task[answer_key]
        if "messages" not in task:
            task["messages"] = [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": str(task.get("question", ""))},
            ]
        tasks.append(task)

    logger.info("Loaded %d tasks from Parquet: %s", len(tasks), path)
    return tasks


def load_jsonl(
    path: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load tasks from a JSONL file.

    Each line should be a JSON object with at least ``question`` and
    ``ground_truth`` fields.

    Args:
        path: Path to the ``.jsonl`` file.
        max_samples: Limit number of samples.

    Returns:
        List of task dicts.
    """
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if "messages" not in task:
                task["messages"] = [
                    {"role": "system", "content": MATH_SYSTEM_PROMPT},
                    {"role": "user", "content": str(task.get("question", task.get("problem", "")))},
                ]
            tasks.append(task)
            if max_samples and len(tasks) >= max_samples:
                break

    logger.info("Loaded %d tasks from JSONL: %s", len(tasks), path)
    return tasks


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K's ``#### <number>`` format."""
    import re

    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip().split("\n")[-1].strip()


def _dummy_math_tasks(n: int, use_tools: bool = False) -> list[dict[str, Any]]:
    """Fallback dummy math tasks when datasets are unavailable."""
    problems = [
        ("What is 7 * 6?", "42"),
        ("What is 15 + 27?", "42"),
        ("If a store has 5 apples and gets 3 more, how many total?", "8"),
        ("What is 100 / 4?", "25"),
        ("A train travels 60 km/h for 2 hours. How far?", "120"),
    ]
    sys_prompt = MATH_TOOL_SYSTEM_PROMPT if use_tools else MATH_SYSTEM_PROMPT
    tasks = []
    for i in range(n):
        q, a = problems[i % len(problems)]
        tasks.append({
            "question": q,
            "ground_truth": a,
            "data_source": "dummy",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ],
        })
    return tasks
