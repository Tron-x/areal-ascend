"""Lightweight tool execution environment for Harbor agents.

Executes tool calls extracted by ``HarborAgentLogic`` without Docker.
Supports:
- ``python`` / ``code_execution``: runs Python code in a subprocess sandbox
- ``bash``: runs shell commands in a subprocess
- Any rllm ``Tool`` registered via ``tool_map``

Used by the multi-turn demo and can be plugged into Forge's
``AgentActor._execute_tools`` flow.

Usage::

    executor = ToolExecutor(timeout=30)
    results = executor.execute(action.tool_calls)
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from typing import Any

from forge.core.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class ToolExecutor:
    """Execute tool calls locally without Docker.

    Args:
        timeout: Max seconds per tool execution.
        python_bin: Python binary path for code execution.
        allowed_tools: If set, only these tool types are allowed.
            None means all tools are allowed.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        python_bin: str = "python3",
        allowed_tools: set[str] | None = None,
    ):
        self._timeout = timeout
        self._python_bin = python_bin
        self._allowed_tools = allowed_tools

    def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute a batch of tool calls and return results."""
        results = []
        for tc in tool_calls:
            if self._allowed_tools and tc.type not in self._allowed_tools:
                results.append(ToolResult(
                    success=False,
                    error=f"Tool '{tc.type}' not allowed",
                    tool_call=tc,
                ))
                continue

            try:
                result = self._dispatch(tc)
            except Exception as e:
                result = ToolResult(success=False, error=str(e), tool_call=tc)
            results.append(result)
        return results

    def _dispatch(self, tc: ToolCall) -> ToolResult:
        """Route a tool call to the appropriate handler."""
        if tc.type in ("python", "code_execution"):
            return self._exec_python(tc)
        elif tc.type == "bash":
            return self._exec_bash(tc)
        elif tc.type in ("str_replace_editor", "file_editor"):
            return ToolResult(
                success=False,
                error="File editor tools require a Docker environment",
                tool_call=tc,
            )
        else:
            return self._exec_generic(tc)

    def _exec_python(self, tc: ToolCall) -> ToolResult:
        """Execute Python code in a subprocess."""
        args = tc.metadata.get("rllm_arguments", {})
        if isinstance(args, dict) and "code" in args:
            code = args["code"]
        else:
            try:
                parsed = json.loads(tc.content)
                code = parsed.get("code", tc.content)
            except (json.JSONDecodeError, TypeError):
                code = tc.content

        if not code.strip():
            return ToolResult(success=True, output="(empty code)", tool_call=tc)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [self._python_bin, tmp_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                return ToolResult(success=True, output=output or "(no output)", tool_call=tc)
            else:
                error = result.stderr.strip() or f"Exit code {result.returncode}"
                return ToolResult(success=False, output=result.stdout.strip(), error=error, tool_call=tc)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Timeout after {self._timeout}s", tool_call=tc)
        finally:
            import os
            os.unlink(tmp_path)

    def _exec_bash(self, tc: ToolCall) -> ToolResult:
        """Execute a bash command."""
        args = tc.metadata.get("rllm_arguments", {})
        if isinstance(args, dict) and "command" in args:
            cmd = args["command"]
        else:
            try:
                parsed = json.loads(tc.content)
                cmd = parsed.get("command", tc.content)
            except (json.JSONDecodeError, TypeError):
                cmd = tc.content

        if not cmd.strip():
            return ToolResult(success=True, output="(empty command)", tool_call=tc)

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = result.stdout.strip()
            if result.returncode == 0:
                return ToolResult(success=True, output=output or "(no output)", tool_call=tc)
            else:
                error = result.stderr.strip() or f"Exit code {result.returncode}"
                return ToolResult(success=False, output=output, error=error, tool_call=tc)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Timeout after {self._timeout}s", tool_call=tc)

    def _exec_generic(self, tc: ToolCall) -> ToolResult:
        """Handle unknown tool types by trying rllm MultiTool if available."""
        try:
            from rllm.tools.multi_tool import MultiTool

            mt = MultiTool(tools=[tc.type])
            args = tc.metadata.get("rllm_arguments", {})
            if not isinstance(args, dict):
                try:
                    args = json.loads(tc.content)
                except (json.JSONDecodeError, TypeError):
                    args = {"input": tc.content}
            output = mt.execute(tc.type, **args)
            return ToolResult(
                success=True,
                output=str(output) if output else "(no output)",
                tool_call=tc,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Unknown tool '{tc.type}': {e}",
                tool_call=tc,
            )
