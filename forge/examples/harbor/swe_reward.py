"""SWE-bench remote reward function for Forge.

Wraps ``SWEBenchEvaluator`` as a Forge-compatible ``RewardFn`` so that
SWE-bench Docker evaluation on a remote x86 server can be used as the
reward signal in GRPO training.

Usage::

    from forge.examples.harbor.swe_reward import SWERewardFn

    reward_fn = SWERewardFn(
        host="142.171.20.182",
        task_data_dir="/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks",
    )

    # As a Forge RewardFn:
    score = reward_fn(prompt="...", response="...", task_id="dask__zict-64")
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_patch(response: str) -> str:
    """Extract a diff/patch from the model response.

    Handles both closed and truncated code blocks.
    """
    closed = [
        r"```diff\s*\n(.*?)```",
        r"```patch\s*\n(.*?)```",
        r"```\s*\n(diff --git.*?)```",
        r"```\s*\n(---.*?\+\+\+.*?)```",
    ]
    for pattern in closed:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    truncated = [
        r"```diff\s*\n(.*)",
        r"```patch\s*\n(.*)",
        r"```\s*\n(diff --git.*)",
    ]
    for pattern in truncated:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content and ("diff --git" in content or "---" in content):
                return content

    return ""


class SWERewardFn:
    """Forge-compatible reward function using remote SWE-bench Docker evaluation.

    Satisfies the ``RewardFn`` protocol: ``(prompt, response, target, **kwargs) -> float``

    Args:
        host: x86 server hostname.
        user: SSH user on x86 server.
        task_data_dir: Local path to Harbor task directories.
        cleanup_after: If True, remove Docker image after evaluation.
        docker_timeout: Max seconds for Docker evaluation.
    """

    def __init__(
        self,
        host: str = "142.171.20.182",
        user: str = "root",
        task_data_dir: str = "/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks",
        cleanup_after: bool = False,
        docker_timeout: int = 600,
    ):
        from forge.examples.harbor.swe_evaluator import SWEBenchEvaluator

        self._evaluator = SWEBenchEvaluator(
            host=host,
            user=user,
            task_data_dir=task_data_dir,
            docker_timeout=docker_timeout,
        )
        self._cleanup = cleanup_after

    def __call__(
        self,
        prompt: str,
        response: str,
        target: Any = None,
        **kwargs: Any,
    ) -> float:
        """Compute reward by running SWE-bench Docker evaluation on x86.

        Args:
            prompt: The task instruction (instruction.md content).
            response: The model's full response containing a patch.
            target: Unused (reward comes from test.sh).
            **kwargs: Must include ``task_id`` (e.g. ``"dask__zict-64"``).

        Returns:
            1.0 if all tests pass, 0.0 otherwise.
        """
        task_id = kwargs.get("task_id", "")
        if not task_id:
            logger.warning("SWERewardFn: no task_id provided, returning 0.0")
            return 0.0

        patch = extract_patch(response)
        if not patch:
            logger.info("SWERewardFn[%s]: no patch extracted, reward=0.0", task_id)
            return 0.0

        result = self._evaluator.evaluate(task_id, patch_text=patch)

        if result.error:
            logger.warning("SWERewardFn[%s]: eval error: %s", task_id, result.error)

        logger.info(
            "SWERewardFn[%s]: reward=%.1f, duration=%.0fs",
            task_id, result.reward, result.duration_sec,
        )

        if self._cleanup:
            self._evaluator.cleanup(task_id)

        return result.reward

    @property
    def evaluator(self):
        return self._evaluator
