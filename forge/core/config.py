"""Forge-native configuration — decoupled from any specific framework's CLI args.

``ForgeConfig`` is the top-level config that forge apps use.
Framework-specific adapters provide ``ConfigBridge`` implementations that
translate external config formats (areal CLI, slime YAML, etc.) into this
unified structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ForgeConfig:
    """Top-level configuration for a Forge training run.

    Attributes:
        experiment_name: Name of the experiment (for logging/checkpoints).
        trial_name: Name of the trial within the experiment.
        run_id: Run identifier (for recovery).

        model_path: HuggingFace model ID or local path (used for tokenizer / chat template).

        train_world_size: Number of training processes (FSDP/DP).
        gen_world_size: Number of inference GPUs.
        master_addr: Distributed master address.
        master_port: Distributed master port.

        reward_fn_path: Dotted import path for the reward function.
        training_script: Path to the training script to execute.
        training_args: CLI args for the training script.

        engine_args: Dict of vLLM/SGLang engine arguments.
        trainer_env: Environment variables for trainer processes.

        backend_type: Adapter backend to use (``"areal"``, ``"slime"``, etc.).
        backend_config: Opaque dict passed to the adapter for backend-specific settings.

        agent_mode: ``"managed"`` (AgentLogic-driven) or ``"cli_native"``
                    (external process via ModelProxyServer HTTP API).
        external_agent_command: Command for CLI-Native mode (e.g. ``["python", "agent.py"]``).
        external_agent_timeout: Max seconds for external agent process.

        async_pipeline: If True, run rollout and training as concurrent loops
                        connected via ReplayBuffer for ~2x GPU utilization.
        replay_buffer_size: Maximum entries in the ReplayBuffer.
        max_staleness_steps: Max age (in training steps) before buffer entries
                             are evicted.  Controls how off-policy data can be.
    """

    experiment_name: str = ""
    trial_name: str = ""
    run_id: int = 0

    model_path: str = ""

    train_world_size: int = 4
    gen_world_size: int = 4
    master_addr: str = ""
    master_port: int = 0

    reward_fn_path: str = ""
    training_script: str = ""
    training_args: list[str] = field(default_factory=list)

    engine_args: dict[str, Any] = field(default_factory=dict)
    trainer_env: dict[str, str] = field(default_factory=dict)

    backend_type: str = "areal"
    backend_config: dict[str, Any] = field(default_factory=dict)

    agent_mode: str = "managed"
    external_agent_command: list[str] = field(default_factory=list)
    external_agent_timeout: float = 300.0

    async_pipeline: bool = False
    replay_buffer_size: int = 4096
    max_staleness_steps: int = 2

    fileroot: str = "/tmp/forge"
    log_dir: str = ""

    def resolve_log_dir(self) -> str:
        """Compute log directory from fileroot/experiment/trial if not set."""
        if self.log_dir:
            return self.log_dir
        import os

        user = os.environ.get("USER", "root")
        return os.path.join(
            self.fileroot,
            "logs",
            user,
            self.experiment_name,
            self.trial_name,
        )
