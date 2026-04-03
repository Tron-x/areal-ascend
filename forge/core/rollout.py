"""Rollout function API and ForgeApp for Forge.

Provides:
- ``RolloutContext``: the context object passed to user rollout functions.
- ``ForgeApp``: the single user-facing entry point.
- ``rollout_batch()``: run a rollout function over a batch of samples.

Inspired by Slime's ``--rollout-function-path`` pattern — the user writes
a single ``async def`` and Forge handles everything else.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from forge.api.config import AppConfig
from forge.api.engine import GenerateEngine
from forge.api.reward import RewardFn
from forge.api.tools import ToolRegistry
from forge.api.types import Sample, SamplingParams

logger = logging.getLogger("forge.rollout")

RolloutFnType = Callable[["Sample", "RolloutContext"], Any]


@dataclass
class RolloutContext:
    """Context object passed to user-defined rollout functions.

    Provides access to the generation engine, tools, reward function,
    and default sampling parameters.
    """

    engine: GenerateEngine
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    reward_fn: RewardFn | None = None
    tokenizer: Any = None
    default_params: SamplingParams = field(default_factory=SamplingParams)
    metadata: dict[str, Any] = field(default_factory=dict)


async def rollout_batch(
    fn: RolloutFnType,
    ctx: RolloutContext,
    samples: list[Sample],
    concurrency: int = 64,
) -> list[Sample]:
    """Run a rollout function over a batch of samples with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    errors: list[Exception] = []

    async def _run_one(sample: Sample) -> Sample:
        async with semaphore:
            try:
                return await fn(sample, ctx)
            except Exception as e:
                logger.error("Rollout failed for sample: %s", e)
                errors.append(e)
                return sample

    tasks = [asyncio.create_task(_run_one(s)) for s in samples]
    results = await asyncio.gather(*tasks)

    if errors:
        logger.warning("%d/%d rollout calls failed", len(errors), len(samples))

    return list(results)


def load_rollout_fn(import_path: str) -> RolloutFnType:
    """Load a rollout function from a dotted import path.

    Example: ``"my_project.rollout.gsm8k_rollout"``
    """
    parts = import_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid rollout_fn path '{import_path}'. "
            "Expected 'module.path.function_name'."
        )
    module_path, fn_name = parts
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"{import_path} is not callable")
    return fn


class ForgeApp:
    """Single user-facing entry point for Forge training.

    Usage::

        app = ForgeApp()

        @app.rollout_fn
        async def my_rollout(sample: Sample, ctx: RolloutContext) -> Sample:
            result = await ctx.engine.generate([sample.prompt], ctx.default_params)
            reward = compute_reward(sample.label, result[0].text)
            return sample.with_response(result[0].text, reward)

        app.run(AppConfig(model="Qwen/Qwen2.5-1.5B", train_gpus=4, infer_gpus=4))

    Internally delegates to the Monarch adapter for actor creation
    and ``forge.apps.grpo.run_pipeline()`` for the training loop.
    """

    def __init__(self) -> None:
        self._rollout_fn: RolloutFnType | None = None
        self._tools = ToolRegistry()

    def rollout_fn(self, fn: RolloutFnType) -> RolloutFnType:
        """Decorator to register the rollout function."""
        self._rollout_fn = fn
        return fn

    def register_tool(self, tool: Any) -> None:
        """Register a tool for use in rollout functions."""
        self._tools.register(tool)

    def run(self, config: AppConfig) -> None:
        """Launch the full training pipeline.

        1. Resolve the rollout function (decorator or config path).
        2. Set up Monarch actors via the adapter layer.
        3. Run the generic pipeline loop.
        """
        if self._rollout_fn is None and config.rollout_fn_path is not None:
            self._rollout_fn = load_rollout_fn(config.rollout_fn_path)

        asyncio.run(self._run_async(config))

    async def _run_async(self, config: AppConfig) -> None:
        from forge.adapters.monarch.setup import setup_grpo
        from forge.apps.grpo import run_pipeline

        ctx = await setup_grpo(config, run_id=0)

        await run_pipeline(
            rollout=ctx["rollout"],
            train=ctx["train"],
            buffer=ctx["buffer"],
            max_steps=ctx["max_steps"],
            start_step=ctx["start_step"],
        )
