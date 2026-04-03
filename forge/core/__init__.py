"""Forge core — framework-agnostic protocols, types, and configuration.

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
from forge.core.protocols import InferenceBridge, RewardBackend, TrainBackend

__all__ = [
    "AgentAction",
    "AgentLogic",
    "CHATML",
    "ChatTemplate",
    "GenerationResult",
    "InferenceBridge",
    "LLAMA_STYLE",
    "RewardBackend",
    "SimpleChatTemplate",
    "ToolCall",
    "ToolResult",
    "TrainBackend",
]
