"""Forge: Framework-Agnostic Agentic RL Plugin.

Typed protocols and a simple user API for building RL training
pipelines on Monarch.

Two user entry styles:

1. Simple: ``@app.rollout_fn`` decorator for defining rollout logic.
2. Advanced: subclass ``ForgeActor`` for custom distributed actors.

Quick start::

    from forge import ForgeApp, Sample, RolloutContext, AppConfig

    app = ForgeApp()

    @app.rollout_fn
    async def my_rollout(sample: Sample, ctx: RolloutContext) -> Sample:
        result = await ctx.engine.generate([sample.prompt], ctx.default_params)
        reward = compute_reward(sample.label, result[0].text)
        return sample.with_response(result[0].text, reward)

    app.run(AppConfig(model="Qwen/Qwen2.5-1.5B", train_gpus=4, infer_gpus=4))
"""

from forge.api.config import AppConfig, ProcessConfig, ServiceConfig
from forge.api.engine import GenerateEngine, RolloutStage, TrainStage
from forge.api.reward import RewardFn
from forge.api.tools import Tool, ToolRegistry
from forge.api.types import GenerateResult, Metrics, Sample, SamplingParams, TrainBatch
from forge.core.actor import ActorConfig, ForgeActor
from forge.core.rollout import ForgeApp, RolloutContext

__all__ = [
    "ActorConfig",
    "AppConfig",
    "ForgeActor",
    "ForgeApp",
    "GenerateEngine",
    "GenerateResult",
    "Metrics",
    "ProcessConfig",
    "RewardFn",
    "RolloutContext",
    "RolloutStage",
    "Sample",
    "SamplingParams",
    "ServiceConfig",
    "Tool",
    "ToolRegistry",
    "TrainBatch",
    "TrainStage",
]
