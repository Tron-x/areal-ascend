"""Tool registry -- register, discover, and execute tools by name.

Inspired by ROLL's ``register_tools`` / ``make_tool`` with dynamic
import via entry_point strings, combined with Slime's ``ToolRegistry``
for OpenAI-spec generation.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from forge.tools.protocol import Tool, ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for tool discovery and execution.

    Usage::

        registry = ToolRegistry()

        # Register via ToolSpec
        registry.register(ToolSpec(
            name="code_interpreter",
            description="Execute Python code",
            parameters={"type":"object","properties":{"code":{"type":"string"}}},
            entry_point="forge.tools.python_sandbox:PythonTool",
        ))

        # Register a Tool instance directly
        registry.register_tool(my_tool_instance)

        # Get OpenAI-compatible specs for prompt injection
        specs = registry.get_tool_specs()

        # Execute a parsed tool call
        result = await registry.execute(ToolCall(name="code_interpreter", arguments={"code":"print(1)"}))
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._instances: dict[str, Tool] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool by specification.

        If ``spec.entry_point`` is set (e.g. ``"module:ClassName"``),
        the tool class will be lazily imported on first ``execute()``.
        """
        self._specs[spec.name] = spec
        logger.debug(f"Registered tool spec: {spec.name}")

    def register_tool(self, tool: Tool) -> None:
        """Register a live Tool instance."""
        self._specs[tool.name] = tool.spec
        self._instances[tool.name] = tool
        logger.debug(f"Registered tool instance: {tool.name}")

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return all tool specs in OpenAI function-calling format.

        Suitable for injecting into the system prompt.
        """
        return [spec.to_openai_spec() for spec in self._specs.values()]

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def list_tools(self) -> list[str]:
        return list(self._specs.keys())

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, resolving the tool instance if needed.

        Args:
            call: Parsed tool call with name and arguments.

        Returns:
            ``ToolResult`` with success/output/error.
        """
        if call.name not in self._specs:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {call.name!r}. Available: {self.list_tools()}",
                tool_call=call,
            )

        tool = self._resolve_tool(call.name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Could not instantiate tool: {call.name!r}",
                tool_call=call,
            )

        try:
            result = await tool.execute(call.arguments)
            result.tool_call = call
            return result
        except Exception as e:
            logger.warning(f"Tool {call.name} execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                tool_call=call,
            )

    def _resolve_tool(self, name: str) -> Tool | None:
        """Get or lazily create a Tool instance."""
        if name in self._instances:
            return self._instances[name]

        spec = self._specs.get(name)
        if spec is None or spec.entry_point is None:
            return None

        try:
            tool = _import_and_create(spec.entry_point)
            self._instances[name] = tool
            return tool
        except Exception as e:
            logger.error(f"Failed to import tool {name} from {spec.entry_point}: {e}")
            return None


def _import_and_create(entry_point: str, **kwargs: Any) -> Any:
    """Import a class from a ``module:ClassName`` string and instantiate it."""
    if ":" in entry_point:
        module_path, class_name = entry_point.split(":", 1)
    elif "." in entry_point:
        parts = entry_point.rsplit(".", 1)
        module_path, class_name = parts[0], parts[1]
    else:
        raise ValueError(f"Invalid entry_point format: {entry_point!r}")

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
