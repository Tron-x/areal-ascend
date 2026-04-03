"""Forge API layer: protocols, types, and configuration.

This package has ZERO framework dependencies — only Python stdlib.
All framework-specific code lives in ``forge.adapters``.
"""

from forge.api.config import AppConfig, ProcessConfig, ServiceConfig
from forge.api.engine import GenerateEngine, RolloutStage, TrainStage
from forge.api.reward import RewardFn
from forge.api.tools import Tool, ToolRegistry
from forge.api.types import GenerateResult, Metrics, Sample, SamplingParams, TrainBatch

__all__ = [
    "AppConfig",
    "GenerateEngine",
    "GenerateResult",
    "Metrics",
    "ProcessConfig",
    "RewardFn",
    "RolloutStage",
    "Sample",
    "SamplingParams",
    "ServiceConfig",
    "Tool",
    "ToolRegistry",
    "TrainBatch",
    "TrainStage",
]
