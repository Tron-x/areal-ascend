"""Forge: Framework-Agnostic Agentic RL Plugin.

Provides typed protocols and a simple user API for building RL training
pipelines that can run on Monarch (AReaL), Ray (Slime), or any other
distributed backend.

Two user entry styles:

1. Simple: ``@app.rollout_fn`` decorator for defining rollout logic.
2. Advanced: subclass ``ForgeActor`` for custom distributed actors.

Quick start::

    from forge import ForgeApp, Sample, RolloutContext

    app = ForgeApp()

    @app.rollout_fn
    async def my_rollout(sample: Sample, ctx: RolloutContext) -> Sample:
        result = await ctx.engine.generate(sample.prompt, ctx.default_params)
        reward = compute_reward(sample.label, result.text)
        return sample.with_response(result.text, reward)

    app.run(AppConfig(model="Qwen/Qwen2.5-1.5B", train_gpus=4, infer_gpus=4))
"""

from forge.api.config import AppConfig, ProcessConfig, ServiceConfig
from forge.api.engine import GenerateEngine, GenerateResult, TrainEngine
from forge.api.reward import RewardFn
from forge.api.tools import Tool, ToolRegistry
from forge.api.types import Metrics, Sample, SamplingParams, TrainBatch
from forge.core.actor import ActorConfig, ForgeActor
from forge.core.rollout import ForgeApp, RolloutContext

__all__ = [
    "AppConfig",
    "ActorConfig",
    "ForgeActor",
    "ForgeApp",
    "GenerateEngine",
    "GenerateResult",
    "Metrics",
    "ProcessConfig",
    "RewardFn",
    "RolloutContext",
    "Sample",
    "SamplingParams",
    "ServiceConfig",
    "Tool",
    "ToolRegistry",
    "TrainBatch",
    "TrainEngine",
]
