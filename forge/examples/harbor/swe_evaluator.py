"""Remote SWE-bench evaluator -- runs Docker evaluation on a remote x86 server via SSH.

Designed for the split architecture where:
- NPU server: runs vLLM inference + GRPO training (aarch64)
- x86 server: runs SWE-bench Docker containers for evaluation

Usage::

    evaluator = SWEBenchEvaluator(
        host="142.171.20.182",
        user="root",
        task_data_dir="/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks",
    )

    # Evaluate a single task (pulls image if needed)
    reward = evaluator.evaluate("12rambau__sepal_ui-814", patch_text="...")

    # Cleanup to save disk
    evaluator.cleanup("12rambau__sepal_ui-814")
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

X86_EVAL_DIR = "/tmp/harbor_eval"


@dataclass
class EvalResult:
    task_id: str
    reward: float
    test_output: str = ""
    duration_sec: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SWEBenchEvaluator:
    """Execute SWE-bench Docker evaluation on a remote x86 server.

    All communication happens via SSH (key-based auth).
    The evaluator:
    1. Syncs test files to the remote server
    2. Writes the agent's patch into the container
    3. Runs ``docker run`` with ``test.sh``
    4. Reads ``reward.txt`` (0 or 1)

    Args:
        host: x86 server hostname or IP.
        user: SSH username.
        task_data_dir: Local path to Harbor task directories
            (e.g. ``/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks``).
        ssh_key: Path to SSH private key (None = default).
        docker_timeout: Max seconds for a single Docker evaluation.
    """

    def __init__(
        self,
        host: str,
        user: str = "root",
        task_data_dir: str = "/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks",
        ssh_key: str | None = None,
        docker_timeout: int = 600,
    ):
        self._host = host
        self._user = user
        self._task_data_dir = Path(task_data_dir)
        self._ssh_key = ssh_key
        self._docker_timeout = docker_timeout

    def _ssh_cmd(self, cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        """Run a command on the remote x86 server via SSH."""
        ssh_args = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
        ]
        if self._ssh_key:
            ssh_args.extend(["-i", self._ssh_key])
        ssh_args.append(f"{self._user}@{self._host}")
        ssh_args.append(cmd)

        return subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout or self._docker_timeout + 60,
        )

    def _scp_to_remote(self, local_path: str, remote_path: str) -> bool:
        """Copy a file or directory to the remote server."""
        scp_args = [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10", "-r",
        ]
        if self._ssh_key:
            scp_args.extend(["-i", self._ssh_key])
        scp_args.extend([local_path, f"{self._user}@{self._host}:{remote_path}"])

        result = subprocess.run(scp_args, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("SCP failed: %s", result.stderr.strip())
            return False
        return True

    def _get_task_image(self, task_id: str) -> str:
        """Read the Docker image name from task.toml."""
        toml_path = self._task_data_dir / task_id / "task.toml"
        if not toml_path.exists():
            raise FileNotFoundError(f"task.toml not found: {toml_path}")

        for line in toml_path.read_text().splitlines():
            if line.strip().startswith("docker_image"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise ValueError(f"docker_image not found in {toml_path}")

    def pull_image(self, task_id: str) -> bool:
        """Pull the Docker image for a task on the remote server."""
        image = self._get_task_image(task_id)
        logger.info("Pulling image for %s: %s", task_id, image)
        result = self._ssh_cmd(f"docker pull {image}", timeout=600)
        if result.returncode != 0:
            logger.error("Failed to pull %s: %s", image, result.stderr.strip())
            return False
        logger.info("Image pulled: %s", image)
        return True

    def sync_test_files(self, task_id: str) -> bool:
        """Sync task test files to the remote server."""
        local_tests = self._task_data_dir / task_id / "tests"
        if not local_tests.exists():
            logger.error("Test directory not found: %s", local_tests)
            return False

        remote_dir = f"{X86_EVAL_DIR}/{task_id}"
        self._ssh_cmd(f"mkdir -p {remote_dir}")
        return self._scp_to_remote(str(local_tests), f"{remote_dir}/")

    def evaluate(self, task_id: str, patch_text: str = "") -> EvalResult:
        """Run a full SWE-bench evaluation for one task.

        Args:
            task_id: Task identifier (e.g. ``"12rambau__sepal_ui-814"``).
            patch_text: The agent's code patch to apply inside the container.
                If empty, the evaluation runs without any patch (baseline).

        Returns:
            EvalResult with reward (0.0 or 1.0) and test output.
        """
        t0 = time.time()
        image = self._get_task_image(task_id)
        remote_dir = f"{X86_EVAL_DIR}/{task_id}"

        if not self.sync_test_files(task_id):
            return EvalResult(
                task_id=task_id, reward=0.0,
                error="Failed to sync test files", duration_sec=time.time() - t0,
            )

        check = self._ssh_cmd(f"docker image inspect {image} > /dev/null 2>&1 && echo yes || echo no", timeout=30)
        if "no" in check.stdout:
            if not self.pull_image(task_id):
                return EvalResult(
                    task_id=task_id, reward=0.0,
                    error=f"Failed to pull image {image}", duration_sec=time.time() - t0,
                )

        if patch_text:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(patch_text)
                local_patch = f.name
            try:
                self._scp_to_remote(local_patch, f"{remote_dir}/agent.patch")
            finally:
                os.unlink(local_patch)

            apply_patch_cmd = (
                "if [ -f /eval/agent.patch ] && [ -s /eval/agent.patch ]; then "
                "cd /testbed && git apply --whitespace=nowarn /eval/agent.patch 2>/dev/null || "
                "echo '[warn] patch apply failed'; fi && "
            )
        else:
            apply_patch_cmd = ""

        docker_cmd = (
            f"docker run --rm "
            f"-v {remote_dir}/tests:/tests "
            f"-v {remote_dir}:/eval "
            f"-v /tmp/harbor_logs_{task_id}:/logs "
            f"{image} "
            f"bash -c '{apply_patch_cmd}bash /tests/test.sh'"
        )

        logger.info("Running Docker evaluation for %s ...", task_id)
        result = self._ssh_cmd(docker_cmd, timeout=self._docker_timeout)

        test_output = result.stdout + result.stderr

        reward_cmd = f"cat /tmp/harbor_logs_{task_id}/verifier/reward.txt 2>/dev/null || echo 0"
        reward_result = self._ssh_cmd(reward_cmd, timeout=10)
        try:
            reward = float(reward_result.stdout.strip())
        except (ValueError, TypeError):
            reward = 0.0

        duration = time.time() - t0
        logger.info(
            "Evaluation %s: reward=%.1f, duration=%.0fs",
            task_id, reward, duration,
        )

        return EvalResult(
            task_id=task_id,
            reward=reward,
            test_output=test_output[-2000:] if len(test_output) > 2000 else test_output,
            duration_sec=duration,
            metadata={"image": image, "had_patch": bool(patch_text)},
        )

    def cleanup(self, task_id: str) -> None:
        """Remove Docker image and eval files for a task to free disk."""
        try:
            image = self._get_task_image(task_id)
            self._ssh_cmd(f"docker rmi {image} 2>/dev/null || true", timeout=30)
        except Exception:
            pass
        remote_dir = f"{X86_EVAL_DIR}/{task_id}"
        self._ssh_cmd(f"rm -rf {remote_dir} /tmp/harbor_logs_{task_id}", timeout=10)
        logger.info("Cleaned up %s", task_id)

    def get_remote_disk_usage(self) -> str:
        """Check disk usage on the remote server."""
        result = self._ssh_cmd("df -h / | tail -1 && echo '---' && docker system df 2>/dev/null", timeout=15)
        return result.stdout.strip()

    def list_available_tasks(self, max_tasks: int = 5) -> list[str]:
        """List task IDs available locally, sorted by name length (proxy for simplicity)."""
        if not self._task_data_dir.exists():
            return []
        tasks = sorted(
            [d.name for d in self._task_data_dir.iterdir() if d.is_dir()],
            key=len,
        )
        return tasks[:max_tasks]
