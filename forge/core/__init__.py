"""Forge core -- framework-agnostic protocols, types, and configuration.

This package has ZERO external framework dependencies (no areal, no torch, etc.).
Only stdlib + typing are allowed here.
"""

from forge.core.agent import (
    AgentAction,
    AgentLogic,
    GenerationResult,
    ToolCall,
    ToolResult,
)
from forge.core.chat_template import (
    CHATML,
    LLAMA_STYLE,
    ChatTemplate,
    SimpleChatTemplate,
)
from forge.core.protocols import (
    DataProvider,
    InferenceBridge,
    InferenceEngine,
    RewardBackend,
    RewardFn,
    TrainBackend,
    TrainEngine,
)
from forge.core.types import Completion, Episode, Group, TrainBatch

__all__ = [
    "AgentAction",
    "AgentLogic",
    "CHATML",
    "ChatTemplate",
    "Completion",
    "DataProvider",
    "Episode",
    "GenerationResult",
    "Group",
    "InferenceBridge",
    "InferenceEngine",
    "LLAMA_STYLE",
    "RewardBackend",
    "RewardFn",
    "SimpleChatTemplate",
    "ToolCall",
    "ToolResult",
    "TrainBackend",
    "TrainBatch",
    "TrainEngine",
]
