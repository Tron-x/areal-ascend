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

from areal.monarch_plugin.controller.actor import AReaLForgeActor

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


class SandboxActor(AReaLForgeActor):
    """Monarch Actor providing subprocess-isolated Python execution.

    Deploy as a service for load-balanced code execution::

        sandbox = await SandboxActor.options(
            num_replicas=4, procs=1
        ).as_service()
        result = await sandbox.execute_code.route(code_str)
    """

    procs = 1
    with_gpus = False

    def __init__(self):
        self._call_count = 0
        self._success_count = 0
        self._total_time = 0.0

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
    def get_stats(self) -> dict:
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "success_count": self._success_count,
            "total_time": self._total_time,
            "avg_time": avg,
        }

    @endpoint
    def shutdown(self) -> None:
        logger.info(
            f"[SandboxActor] Shutting down. "
            f"Executed {self._call_count} code blocks "
            f"({self._success_count} succeeded)"
        )
