"""Configuration dataclasses for Forge.

All configs are plain Python dataclasses -- no dependency on Hydra,
OmegaConf, or any framework-specific config system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessConfig:
    """Resource requirements for a single actor or service replica."""

    procs: int = 1
    with_gpus: bool = False
    hosts: int | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    bootstrap: Any = None


@dataclass
class ServiceConfig:
    """Configuration for a multi-replica service."""

    process: ProcessConfig = field(default_factory=ProcessConfig)
    num_replicas: int = 1
    router: str = "least_loaded"
    max_concurrency: int = 16
    health_check_interval: float = 30.0


@dataclass
class WeightSyncConfig:
    """Configuration for weight synchronization between trainer and generator."""

    method: str = "xccl"
    disk_path: str = "/tmp/forge_weights"
    timeout: float = 300.0


@dataclass
class AppConfig:
    """Top-level application configuration.

    This is the single entrypoint for configuring a Forge training run.
    Users can set it directly in Python or load from YAML/CLI.
    """

    model: str = ""
    infer_gpus: int = 4
    train_gpus: int = 4

    backend: str = "monarch"

    max_steps: int = 100
    batch_size: int = 256
    buffer_size: int = 16

    rollout_fn_path: str | None = None

    reward_fn_path: str = "areal.reward.gsm8k.gsm8k_reward_fn"

    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)

    generator: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(
            process=ProcessConfig(procs=1, with_gpus=True),
            num_replicas=1,
        )
    )
    trainer: ProcessConfig = field(
        default_factory=lambda: ProcessConfig(
            procs=4,
            with_gpus=True,
        )
    )

    log_interval: int = 1
    save_interval: int = -1
    eval_interval: int = -1
    checkpoint_dir: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AppConfig:
        """Create from a flat or nested dictionary."""
        weight_sync_data = d.pop("weight_sync", {})
        generator_data = d.pop("generator", {})
        trainer_data = d.pop("trainer", {})

        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

        if weight_sync_data:
            cfg.weight_sync = WeightSyncConfig(**weight_sync_data)
        if generator_data:
            proc = generator_data.pop("process", {})
            cfg.generator = ServiceConfig(
                process=ProcessConfig(**proc),
                **{
                    k: v
                    for k, v in generator_data.items()
                    if k in ServiceConfig.__dataclass_fields__
                },
            )
        if trainer_data:
            cfg.trainer = ProcessConfig(**trainer_data)

        return cfg
