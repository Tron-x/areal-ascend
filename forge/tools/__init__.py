"""Forge tool system -- pluggable tools for agentic RL.

Provides tool registration, action parsing, and execution for
multi-turn tool-integrated reasoning (ReTool / TIR).

Core abstractions:
    ``Tool``          -- protocol for any executable tool
    ``ActionParser``  -- protocol for extracting tool calls from LLM output
    ``ToolRegistry``  -- register/discover/execute tools by name
    ``PythonSandbox`` -- safe Python code execution

Design references:
    - ROLL (Alibaba): ToolSpec + register_tools + ActionParser
    - Slime (MiniMax): ToolRegistry + PythonSandbox + token-level loss mask
    - OpenAI: function calling JSON schema for tool specs
"""

from forge.tools.parsers import (
    CodeBlockParser,
    CompositeParser,
    FunctionCallParser,
    ToolCallParser,
)
from forge.tools.protocol import ActionParser, Tool, ToolCall, ToolResult, ToolSpec
from forge.tools.registry import ToolRegistry

__all__ = [
    "ActionParser",
    "CodeBlockParser",
    "CompositeParser",
    "FunctionCallParser",
    "Tool",
    "ToolCall",
    "ToolCallParser",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]
