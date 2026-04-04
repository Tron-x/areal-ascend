"""Forge core -- framework-agnostic protocols, types, and configuration.

This package has ZERO external framework dependencies (no areal, no torch, etc.).
Only stdlib + typing are allowed here.

Five files:
    - ``types.py``          -- data classes (Episode, TrainBatch, Completion, AgentAction, ...)
    - ``protocols.py``      -- engine + agent protocols (TrainEngine, AgentLogic, ...)
    - ``weight_sync.py``    -- WeightSyncStrategy protocol + data types
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
    RewardModelEngine,
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
from forge.core.weight_sync import (
    WeightSyncConfig,
    WeightSyncMethod,
    WeightSyncStrategy,
    WeightsSpec,
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
    "RewardModelEngine",
    "SimpleChatTemplate",
    "ToolCall",
    "ToolResult",
    "TrainBackend",
    "TrainBatch",
    "TrainEngine",
    "WeightSyncConfig",
    "WeightSyncMethod",
    "WeightSyncStrategy",
    "WeightsSpec",
]
