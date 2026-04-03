"""ExternalAgentRunner -- launch external agent processes for CLI-Native Mode.

In CLI-Native Mode (ROLL-inspired), the agent logic lives in an external
process (Python script, shell command, etc.) that communicates with
Forge purely through the ModelProxy HTTP API.

The runner manages:
1. Starting the external process with the correct environment variables
2. Waiting for process completion (with timeout)
3. Collecting the exit status and any output

The HTTP ModelProxyServer records the trajectory (all LLM calls with
training metadata) while the external process runs.

Usage::

    runner = ExternalAgentRunner(
        command=["python", "my_agent.py"],
        model_proxy_url="http://localhost:8100",
    )
    exit_code = await runner.run(
        env_extra={"TASK_DATA": json.dumps(task)},
        timeout=120.0,
    )

The external agent script uses the standard OpenAI client::

    import openai
    client = openai.OpenAI(
        base_url=os.environ["FORGE_MODEL_PROXY_URL"],
        api_key="not-needed",
    )
    response = client.chat.completions.create(
        model="forge-generator",
        messages=[{"role": "user", "content": "Solve: 2+3=?"}],
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExternalAgentResult:
    """Result of running an external agent process."""

    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration: float = 0.0


class ExternalAgentRunner:
    """Manage the lifecycle of an external agent process.

    The runner injects ``FORGE_MODEL_PROXY_URL`` (and optionally
    ``FORGE_SANDBOX_URL``) into the subprocess environment so the
    external agent can discover the Forge services.

    Args:
        command: Command to run (list of strings, like subprocess).
        model_proxy_url: Base URL of the ModelProxyServer.
        sandbox_url: Optional URL for a sandbox HTTP service.
        working_dir: Working directory for the subprocess.
        extra_env: Additional environment variables to inject.
    """

    def __init__(
        self,
        command: list[str],
        model_proxy_url: str,
        sandbox_url: str | None = None,
        working_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
    ):
        self._command = command
        self._model_proxy_url = model_proxy_url
        self._sandbox_url = sandbox_url
        self._working_dir = working_dir
        self._extra_env = extra_env or {}

    def _build_env(self, env_extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build the subprocess environment."""
        env = os.environ.copy()
        env["FORGE_MODEL_PROXY_URL"] = self._model_proxy_url
        env["OPENAI_BASE_URL"] = f"{self._model_proxy_url}/v1"
        env["OPENAI_API_KEY"] = "forge-internal"
        if self._sandbox_url:
            env["FORGE_SANDBOX_URL"] = self._sandbox_url
        env.update(self._extra_env)
        if env_extra:
            env.update(env_extra)
        return env

    async def run(
        self,
        env_extra: dict[str, str] | None = None,
        timeout: float = 300.0,
        stdin_data: str | None = None,
    ) -> ExternalAgentResult:
        """Launch the external agent and wait for completion.

        Args:
            env_extra: Per-episode environment variables (e.g. task data).
            timeout: Maximum execution time in seconds.
            stdin_data: Optional data to write to the process's stdin.

        Returns:
            ``ExternalAgentResult`` with exit code, stdout, stderr.
        """
        import time

        env = self._build_env(env_extra)
        cmd_str = " ".join(self._command)
        logger.info(f"Launching external agent: {cmd_str}")

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                env=env,
                cwd=self._working_dir,
            )

            stdin_bytes = stdin_data.encode() if stdin_data else None
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=timeout,
            )

            duration = time.monotonic() - t0
            result = ExternalAgentResult(
                exit_code=proc.returncode or 0,
                stdout=stdout_bytes.decode(errors="replace")[-8192:],
                stderr=stderr_bytes.decode(errors="replace")[-8192:],
                duration=duration,
            )
            logger.info(
                f"External agent finished: exit={result.exit_code}, "
                f"duration={duration:.1f}s"
            )
            return result

        except TimeoutError:
            duration = time.monotonic() - t0
            logger.warning(
                f"External agent timed out after {timeout}s, killing process"
            )
            proc.kill()
            await proc.wait()
            stdout_bytes, stderr_bytes = b"", b""
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=5.0
                )
            except (TimeoutError, ProcessLookupError):
                pass
            return ExternalAgentResult(
                exit_code=-1,
                stdout=stdout_bytes.decode(errors="replace")[-8192:],
                stderr=stderr_bytes.decode(errors="replace")[-8192:],
                timed_out=True,
                duration=duration,
            )

        except Exception as e:
            duration = time.monotonic() - t0
            logger.error(f"External agent failed: {e}")
            return ExternalAgentResult(
                exit_code=-1,
                stderr=str(e),
                duration=duration,
            )
