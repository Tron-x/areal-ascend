"""SandboxActor -- subprocess-isolated Python code execution.

Provides a secure execution environment for model-generated code.
Each execution runs in a separate subprocess with a hard timeout.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import time

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)

_RUNNER_TEMPLATE = textwrap.dedent("""\
import sys, json, io
from contextlib import redirect_stdout, redirect_stderr

_stdout = io.StringIO()
_stderr = io.StringIO()
_result = None
_success = True

try:
    _code = {code_repr}
    _globals = {{}}
    with redirect_stdout(_stdout), redirect_stderr(_stderr):
        exec(_code, _globals)
    if "result" in _globals:
        _result = str(_globals["result"])
    elif "answer" in _globals:
        _result = str(_globals["answer"])
except Exception as _e:
    _success = False
    _stderr.write(str(_e))

print(json.dumps({{
    "success": _success,
    "stdout": _stdout.getvalue()[:4096],
    "stderr": _stderr.getvalue()[:4096],
    "result": (_result or _stdout.getvalue().strip())[:4096],
}}))
""")


class SandboxActor(ForgeActor):
    """Monarch Actor providing subprocess-isolated Python execution
    and -- as of A2-full -- a pluggable tool server backed by
    :class:`forge.tools.ToolRegistry`.

    Deploy as a service for load-balanced code execution::

        sandbox = await SandboxActor.options(
            num_replicas=4, procs=1
        ).as_service()
        result = await sandbox.execute_code.route(code_str)

    Multi-tool mode (A2-full)::

        # YAML: roles.tool_server.tools: [python_sandbox, calculator]
        sandbox = await SandboxActor.options(
            num_replicas=2, mesh_name="tool_server"
        ).as_service(tool_types=["python_sandbox", "calculator"])
        result = await sandbox.execute_tool.route(
            "calculator", {"expr": "2+2"},
        )

    ``execute_code`` remains as a backward-compatible shortcut for
    ``execute_tool("python_sandbox", {"code": code})`` so existing
    callers keep working with zero changes.
    """

    procs = 1
    with_gpus = False

    def __init__(self, tool_types: list[str] | tuple[str, ...] | None = None):
        """
        Args:
            tool_types: Short names (as registered via
                ``@forge.tools.register_tool``) to mount in this
                actor's :class:`ToolRegistry`.  ``None`` / empty
                keeps legacy behavior (``execute_code`` only, no
                registry).  Resolution is lazy-in-setup so a
                missing tool only raises when the actor tries to
                instantiate it, not during ``__init__`` -- that
                matches Monarch's "construct cheap, init
                expensive" split.
        """
        self._call_count = 0
        self._success_count = 0
        self._total_time = 0.0
        self._tool_types: tuple[str, ...] = tuple(tool_types) if tool_types else ()
        # Built lazily in setup() so construction stays side-effect free.
        self._registry = None

    def _ensure_registry(self):
        """Instantiate the per-actor :class:`ToolRegistry` on demand.

        Built in the actor process (not the driver) so each tool's
        state (e.g. ``PythonSandbox.timeout``) belongs to the worker
        proc and survives restarts independently.
        """
        if self._registry is not None:
            return self._registry
        from forge.tools import ToolRegistry, get_tool

        registry = ToolRegistry()
        for name in self._tool_types:
            cls = get_tool(name)
            registry.register_tool(cls())
        self._registry = registry
        logger.info(
            "[SandboxActor] mounted tools=%s (registry size=%d)",
            list(self._tool_types),
            len(registry.list_tools()),
        )
        return registry

    @endpoint
    def execute_code(self, code: str, timeout: float = 10.0) -> dict:
        """Execute Python code in an isolated subprocess.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with keys: success, stdout, stderr, result.
        """
        t0 = time.monotonic()
        self._call_count += 1

        runner_code = _RUNNER_TEMPLATE.format(code_repr=repr(code))

        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner_code],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=None,
            )
            try:
                result = json.loads(proc.stdout.strip())
            except (json.JSONDecodeError, ValueError):
                result = {
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout[:4096],
                    "stderr": proc.stderr[:4096],
                    "result": proc.stdout.strip()[:4096],
                }
        except subprocess.TimeoutExpired:
            result = {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "result": "",
            }
        except Exception as e:
            result = {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "result": "",
            }

        elapsed = time.monotonic() - t0
        self._total_time += elapsed
        if result.get("success"):
            self._success_count += 1

        return result

    @endpoint
    async def execute_tool(self, name: str, arguments: dict) -> dict:
        """Dispatch a tool call through the mounted
        :class:`ToolRegistry` and return a plain dict so the result
        is trivially Monarch-serializable.

        This is the multi-tool path (A2-full).  ``execute_code`` is
        the python-only shortcut kept for backward compat with
        existing ``AgentActor._call_sandbox`` callers.

        Raises ``RuntimeError`` if no tools were mounted (i.e. the
        actor was constructed with ``tool_types=None``).
        """
        from forge.tools import ToolCall

        if not self._tool_types:
            return {
                "success": False,
                "output": "",
                "error": (
                    "execute_tool called on SandboxActor with no "
                    "tool_types configured; pass "
                    "tool_types=[...] at construction or use "
                    "execute_code directly."
                ),
            }
        registry = self._ensure_registry()
        t0 = time.monotonic()
        self._call_count += 1
        result = await registry.execute(ToolCall(name=name, arguments=arguments))
        elapsed = time.monotonic() - t0
        self._total_time += elapsed
        if result.success:
            self._success_count += 1
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    @endpoint
    def get_stats(self) -> dict:
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "success_count": self._success_count,
            "total_time": self._total_time,
            "avg_time": avg,
            "tool_types": list(self._tool_types),
        }

    @endpoint
    def shutdown(self) -> None:
        logger.info(
            f"[SandboxActor] Shutting down. "
            f"Executed {self._call_count} code blocks "
            f"({self._success_count} succeeded)"
        )
