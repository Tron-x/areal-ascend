"""Forge core -- framework-agnostic protocols, types, and configuration.

This package has ZERO external framework dependencies (no areal, no torch, etc.).
Only stdlib + typing are allowed here.

Four files:
    - ``types.py``          -- data classes (Episode, TrainBatch, Completion, AgentAction, ...)
    - ``protocols.py``      -- engine + agent protocols (TrainEngine, AgentLogic, ...)
    - ``chat_template.py``  -- ChatTemplate protocol + presets
    - ``config.py``         -- ForgeConfig
"""

from forge.core.chat_template import (
    CHATML,
    LLAMA_STYLE,
    ChatTemplate,
    SimpleChatTemplate,
)
from forge.core.protocols import (
    AgentLogic,
    BatchAdapter,
    DataProvider,
    InferenceBridge,
    InferenceEngine,
    RewardBackend,
    RewardFn,
    TrainBackend,
    TrainEngine,
)
from forge.core.types import (
    AgentAction,
    Completion,
    Episode,
    GenerationResult,
    Group,
    ToolCall,
    ToolResult,
    TrainBatch,
)

__all__ = [
    "AgentAction",
    "AgentLogic",
    "BatchAdapter",
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
