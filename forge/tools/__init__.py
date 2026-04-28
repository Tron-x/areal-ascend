"""Forge tool system -- pluggable tools for agentic RL.

Provides tool registration, action parsing, and execution for
multi-turn tool-integrated reasoning (ReTool / TIR).

Core abstractions:
    ``Tool``          -- protocol for any executable tool
    ``ActionParser``  -- protocol for extracting tool calls from LLM output
    ``ToolRegistry``  -- per-actor in-memory registry (register +
                         execute tools by name)
    ``register_tool`` -- module-level decorator for global short-name
                         discovery (mirrors ``forge.reward`` and
                         ``forge.agents``).  Set of tools a
                         ``tool_server`` actor will mount is declared
                         in YAML via ``roles.tool_server.tools:``.

Design references:
    - ROLL (Alibaba): ToolSpec + register_tools + ActionParser
    - Slime (MiniMax): ToolRegistry + PythonSandbox + token-level loss mask
    - OpenAI: function calling JSON schema for tool specs

Registry pattern (A2-full):

    # my_custom_tool.py
    from forge.tools import register_tool, Tool, ToolSpec, ToolResult

    @register_tool("calculator")
    class CalculatorTool:
        name = "calculator"
        spec = ToolSpec(name="calculator", description="Eval math")
        async def execute(self, arguments):
            return ToolResult(success=True, output=str(eval(arguments["expr"])))

    # YAML
    roles:
      tool_server:
        tools: [python_sandbox, calculator]

The ``SandboxActor`` (tool server actor) resolves each short name via
``get_tool(name)`` at construction time and mounts the resulting
class into a per-actor :class:`ToolRegistry` instance.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Any

from forge.tools.parsers import (
    CodeBlockParser,
    CompositeParser,
    FunctionCallParser,
    ToolCallParser,
)
from forge.tools.protocol import ActionParser, Tool, ToolCall, ToolResult, ToolSpec
from forge.tools.registry import ToolRegistry

logger = logging.getLogger("ToolRegistryModule")

# Global short-name -> tool class (or factory).  Anything callable
# that returns a :class:`Tool`-compatible instance is acceptable.
# Instance construction happens inside ``SandboxActor`` (or any other
# tool-server actor) so each actor gets its own per-process state.
_REGISTRY: dict[str, Callable[..., Any]] = {}

_AUTO_DISCOVER_DONE = False


def register_tool(name: str):
    """Decorator: register ``cls`` under ``name`` in the tool registry.

    Mirrors :func:`forge.agents.register_agent` /
    :func:`areal.workflow.register_workflow` / :func:`forge.reward.register_reward`
    so the UX is consistent across registries.

    Duplicates raise ``ValueError`` at decoration time.  If you
    intentionally want to replace an existing entry, delete from
    ``_REGISTRY`` first (or call :func:`reset_for_tests`).
    """

    def wrap(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"tool {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}; "
                f"refusing to overwrite with "
                f"{cls.__module__}.{cls.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_tool(name: str) -> Callable[..., Any]:
    """Lookup a registered tool class by short name.

    Raises ``ValueError`` listing available names + placement hints
    when the short name is unknown.
    """
    ensure_loaded()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"unknown tool {name!r}\n"
            f"\n"
            f"Available tools: {available}\n"
            f"\n"
            f"If you wrote `@register_tool({name!r})` but it's not\n"
            f"listed above, the decorator's module wasn't imported.\n"
            f"Pick ONE of:\n"
            f"\n"
            f"  (a) Auto-discovery (recommended).  Place the file at:\n"
            f"        forge/tools/<anything>.py\n"
            f"      Auto-discovery scans that package at lookup time,\n"
            f"      so a plain file copy is enough.\n"
            f"\n"
            f"  (b) Explicit import from your launch entry script.\n"
            f"      Add before training starts:\n"
            f"        import my_tool_module  # noqa: F401\n"
            f"\n"
            f"Run `python -m forge list-tools` to verify registration."
        )
    return _REGISTRY[name]


def available_tools() -> list[str]:
    """Return the sorted list of registered tool short names."""
    ensure_loaded()
    return sorted(_REGISTRY)


def ensure_loaded() -> None:
    """Import built-in ``forge.tools`` submodules so ``@register_tool``
    decorators fire.

    Idempotent; safe to call from any hot path.
    """
    global _AUTO_DISCOVER_DONE
    if _AUTO_DISCOVER_DONE:
        return
    _AUTO_DISCOVER_DONE = True

    try:
        pkg = importlib.import_module("forge.tools")
    except Exception as e:  # pragma: no cover -- defensive
        logger.debug("ensure_loaded: cannot import forge.tools (%s); skipping", e)
        return
    if not hasattr(pkg, "__path__"):
        return
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        if modinfo.name.startswith("_"):
            continue
        # Skip modules we already imported at the top of this file;
        # pkgutil will happily re-import them but that's a no-op.
        full_name = f"forge.tools.{modinfo.name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.debug("ensure_loaded: cannot import %s (%s); skipping", full_name, e)


def reset_for_tests() -> None:
    """Test-only: clear the registry + re-run discovery on next call."""
    global _AUTO_DISCOVER_DONE
    _REGISTRY.clear()
    _AUTO_DISCOVER_DONE = False


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
    "register_tool",
    "get_tool",
    "available_tools",
    "ensure_loaded",
    "reset_for_tests",
]
