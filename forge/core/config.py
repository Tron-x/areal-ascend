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
    """

    experiment_name: str = ""
    trial_name: str = ""
    run_id: int = 0

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
