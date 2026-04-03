"""Framework-agnostic Sandbox for code execution.

Pure Python subprocess-based execution -- no Monarch or Ray dependency.
Extracted from ``areal/monarch_plugin/sandbox_actor.py``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import time
from typing import Any

logger = logging.getLogger("forge.sandbox")

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


class Sandbox:
    """Subprocess-isolated Python code execution.

    Each ``execute()`` call spawns a fresh Python subprocess with
    a hard timeout, preventing resource leaks and infinite loops.

    Satisfies the ``Tool`` protocol for use in ``ToolRegistry``.
    """

    def __init__(self, default_timeout: float = 10.0) -> None:
        self._default_timeout = default_timeout
        self._call_count = 0
        self._success_count = 0
        self._total_time = 0.0

    @property
    def name(self) -> str:
        return "sandbox"

    @property
    def description(self) -> str:
        return "Execute Python code in an isolated subprocess with a timeout."

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute Python code.

        Parameters (via kwargs)
        -----------------------
        code : str
            Python source code to execute.
        timeout : float, optional
            Maximum execution time in seconds.
        """
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", self._default_timeout)
        return self.execute_sync(code, timeout)

    def execute_sync(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Synchronous code execution in a subprocess."""
        if timeout is None:
            timeout = self._default_timeout

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

    def stats(self) -> dict:
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "success_count": self._success_count,
            "total_time": self._total_time,
            "avg_time": avg,
        }
