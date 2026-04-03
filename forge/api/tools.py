"""Tool protocols for Forge agentic workflows.

Tools are capabilities available to the model during multi-turn rollouts:
sandboxes for code execution, search engines, API callers, etc.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Protocol for a single tool available during rollout."""

    @property
    def name(self) -> str:
        """Unique identifier for this tool (e.g. ``"sandbox"``, ``"search"``)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for the tool."""
        ...

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the given arguments.

        Returns
        -------
        dict
            Must include ``"success"`` (bool) and ``"result"`` (str).
        """
        ...


class ToolRegistry:
    """Registry of tools available during rollout.

    Access tools by attribute (``ctx.tools.sandbox``) or by name
    (``ctx.tools["sandbox"]``).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its ``name`` property."""
        self._tools[tool.name] = tool

    def __getattr__(self, name: str) -> Tool:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._tools[name]
        except KeyError:
            raise AttributeError(
                f"No tool named '{name}'. Available: {list(self._tools)}"
            ) from None

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools)

    def get_specs(self) -> list[dict[str, str]]:
        """Return tool specifications suitable for prompt injection."""
        return [
            {"name": t.name, "description": t.description} for t in self._tools.values()
        ]
