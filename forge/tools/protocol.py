"""Tool protocols and data types.

Zero external dependencies -- pure stdlib + typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolSpec:
    """OpenAI function-calling compatible tool specification.

    Used for:
    1. Registering tools in ``ToolRegistry``
    2. Generating tool prompt instructions for the LLM
    3. Validating tool call arguments

    Format follows OpenAI's tool schema::

        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute Python code",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]
                }
            }
        }
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    entry_point: str | None = None

    def to_openai_spec(self) -> dict[str, Any]:
        """Convert to OpenAI function calling JSON format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A parsed tool invocation from LLM output.

    Attributes:
        name: Tool name (e.g. ``"code_interpreter"``).
        arguments: Tool arguments dict.
        raw_text: The raw text segment that was parsed into this call.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class ToolResult:
    """Result of executing a tool call.

    Attributes:
        success: Whether execution completed without errors.
        output: Tool stdout / return value.
        error: Error message if not successful.
        tool_call: The original call that produced this result.
    """

    success: bool
    output: str = ""
    error: str = ""
    tool_call: ToolCall | None = None


@runtime_checkable
class Tool(Protocol):
    """Protocol for executable tools.

    Implementations must provide:
    - ``name``: unique identifier
    - ``spec``: OpenAI-compatible tool specification
    - ``execute``: run the tool with given arguments
    """

    @property
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    def spec(self) -> ToolSpec:
        """Tool specification for LLM prompt injection."""
        ...

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            arguments: Tool-specific arguments dict.

        Returns:
            ``ToolResult`` with success status and output.
        """
        ...


@runtime_checkable
class ActionParser(Protocol):
    """Protocol for parsing LLM output into tool calls.

    Different models use different formats:
    - ``<code>...</code>`` (ReTool/veRL)
    - ``<tool_call>{"name":...}</tool_call>`` (Qwen3)
    - `` ```python ... ``` `` (markdown code blocks)
    - ``<function=name>...</function>`` (Qwen3 Coder)

    Implementations extract ``ToolCall`` objects from raw text.
    """

    def parse(self, response: str) -> list[ToolCall]:
        """Parse LLM response text into a list of tool calls.

        Returns empty list if no tool calls found.
        """
        ...

    def has_tool_call(self, response: str) -> bool:
        """Quick check if the response contains any tool call markers."""
        ...
