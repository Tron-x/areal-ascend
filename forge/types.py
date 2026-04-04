"""Core type definitions for the AReaL Monarch plugin.

Mirrors TorchForge's type system with AReaL-specific additions for
Ascend NPU support and RL training workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class ProcessConfig:
    """Configuration for allocating a Monarch ProcMesh.

    Args:
        procs: Number of processes to spawn per replica.
        with_gpus: Whether to allocate accelerator devices (NPU/GPU).
        hosts: Number of hosts. None means local-only.
        mesh_name: Unique name for the ProcMesh allocation.
    """

    procs: int = 1
    with_gpus: bool = False
    hosts: int | None = None
    mesh_name: str | None = None


@dataclass
class ServiceConfig:
    """Configuration for a replicated Forge-style service.

    Args:
        procs: Number of processes per replica.
        num_replicas: Number of replicas to maintain.
        with_gpus: Whether each replica needs accelerator devices.
        hosts: Number of hosts per replica. None means local-only.
        health_poll_rate: Seconds between health checks.
        replica_max_concurrent_requests: Max in-flight requests per replica.
        return_first_rank_result: Auto-unwrap ValueMesh to rank-0 result.
        mesh_name: Base name for ProcMesh allocations.
    """

    procs: int
    num_replicas: int
    with_gpus: bool = False
    hosts: int | None = None
    health_poll_rate: float = 0.2
    replica_max_concurrent_requests: int = 10
    return_first_rank_result: bool = True
    mesh_name: str | None = None

    def to_process_config(self) -> ProcessConfig:
        return ProcessConfig(
            procs=self.procs,
            with_gpus=self.with_gpus,
            hosts=self.hosts,
            mesh_name=self.mesh_name,
        )


class Launcher(Enum):
    LOCAL = "local"
    SLURM = "slurm"
    PREALLOCATED = "preallocated"


@dataclass
class LauncherConfig:
    """Cluster launcher configuration.

    Modes:
        - ``local``: All actors on the current machine (default).
        - ``slurm``: Allocate machines via Slurm sbatch.
        - ``preallocated``: Machines already allocated by K8s / external
          scheduler.  Discovered via MASTER_ADDR + NNODES environment
          variables (torchrun-style).
    """

    launcher: Launcher = Launcher.LOCAL
    job_name: str = ""
    services: dict[str, ServiceConfig] = field(default_factory=dict)
    actors: dict[str, ProcessConfig] = field(default_factory=dict)
    gpus_per_node: int = 8
    master_addr: str = ""
    master_port: int = 0
    nnodes: int = 1
    node_rank: int = 0

    def __post_init__(self):
        if isinstance(self.launcher, str):
            self.launcher = Launcher(self.launcher)


@dataclass
class ProvisionerConfig:
    """Configuration for the global resource provisioner."""

    launcher_config: LauncherConfig | None = None


@dataclass
class TrainBatch:
    """Universal training batch for all training modes.

    Usage::

        logits = model(**batch.model_inputs)
        loss = loss_fn(logits, **batch.loss_inputs)

    Attributes:
        model_inputs: Inputs for the model forward pass.
        loss_inputs: Inputs for loss computation (targets, advantages, etc.).
        meta: Extra metadata not used in forward/loss.
    """

    model_inputs: dict[str, Any]
    loss_inputs: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


Scalar = int | float
