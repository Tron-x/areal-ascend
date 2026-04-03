"""Python code sandbox and tool -- safe code execution for agentic RL.

Provides:
- ``PythonSandbox``: subprocess-isolated code execution with timeout,
  safety checks, and memory management (Slime-inspired).
- ``PythonTool``: ``Tool`` protocol implementation wrapping the sandbox.

The sandbox runs user code in a fresh subprocess to prevent:
- Side effects on the training process
- Resource leaks (memory, file handles)
- Security issues (file system access, network, etc.)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import Any

from forge.tools.protocol import ToolResult, ToolSpec

logger = logging.getLogger(__name__)

_DANGEROUS_PATTERNS = [
    r"import\s+os\b",
    r"import\s+sys\b",
    r"import\s+subprocess\b",
    r"import\s+shutil\b",
    r"__import__",
    r"\bopen\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
]

_RUNNER_TEMPLATE = """\
import sys, traceback
from io import StringIO

old_stdout, old_stderr = sys.stdout, sys.stderr
stdout_buf, stderr_buf = StringIO(), StringIO()
sys.stdout, sys.stderr = stdout_buf, stderr_buf

try:
{indented_code}
except Exception:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

out = stdout_buf.getvalue()
err = stderr_buf.getvalue()
if out:
    print(out, end='')
if err:
    print(err, end='', file=sys.stderr)
"""


class PythonSandbox:
    """Subprocess-isolated Python code execution.

    Args:
        timeout: Max execution time in seconds.
        max_output_len: Truncate stdout/stderr beyond this length.
        safety_check: If True, scan code for dangerous patterns.
        allowed_modules: Set of module names allowed for import.
            Only checked if ``safety_check=True``. Pass ``None`` to skip.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_output_len: int = 4096,
        safety_check: bool = True,
        allowed_modules: set[str] | None = None,
    ):
        self.timeout = timeout
        self.max_output_len = max_output_len
        self.safety_check = safety_check
        self.allowed_modules = allowed_modules

    def check_safety(self, code: str) -> tuple[bool, str]:
        """Scan code for dangerous patterns.

        Returns:
            ``(is_safe, message)`` tuple.
        """
        for pattern in _DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked pattern: {pattern}"

        if self.allowed_modules is not None:
            for m in re.findall(r"import\s+(\w+)", code):
                if m not in self.allowed_modules:
                    return False, f"Module not allowed: {m}"
            for m in re.findall(r"from\s+(\w+)", code):
                if m not in self.allowed_modules:
                    return False, f"Module not allowed: {m}"

        return True, "OK"

    async def execute(self, code: str) -> tuple[bool, str, str]:
        """Execute code in a subprocess.

        Returns:
            ``(success, stdout, stderr)`` tuple.
        """
        if self.safety_check:
            is_safe, msg = self.check_safety(code)
            if not is_safe:
                return False, "", f"Safety check failed: {msg}"

        indented = "\n".join("    " + line for line in code.split("\n"))
        script = _RUNNER_TEMPLATE.format(indented_code=indented)

        tmp_dir = tempfile.mkdtemp(prefix="forge_sandbox_")
        script_path = os.path.join(tmp_dir, "code.py")

        try:
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={"PATH": os.environ.get("PATH", "/usr/bin"), "PYTHONPATH": ""},
                cwd=tmp_dir,
            )

            stdout = proc.stdout[: self.max_output_len]
            stderr = proc.stderr[: self.max_output_len]
            return proc.returncode == 0, stdout, stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Timed out after {self.timeout}s"
        except Exception as e:
            return False, "", str(e)
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


class PythonTool:
    """``Tool`` protocol implementation for Python code execution.

    Wraps ``PythonSandbox`` and provides the standard tool interface.

    Usage::

        tool = PythonTool(timeout=30)
        result = await tool.execute({"code": "print(2+2)"})
        # result.output == "4\\n"
    """

    def __init__(self, timeout: float = 30.0, safety_check: bool = True):
        self._sandbox = PythonSandbox(timeout=timeout, safety_check=safety_check)

    @property
    def name(self) -> str:
        return "code_interpreter"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter",
            description=(
                "Execute Python code in a safe sandbox. "
                "Results are captured from print() statements. "
                "Each execution is independent (no state persistence)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code", "")
        if not code.strip():
            return ToolResult(success=False, error="No code provided")

        success, stdout, stderr = await self._sandbox.execute(code)

        if success:
            output = stdout.strip() or "(no output)"
            return ToolResult(success=True, output=output)

        error_msg = stderr.strip() if stderr else "Execution failed"
        return ToolResult(success=False, output=stdout.strip(), error=error_msg)
