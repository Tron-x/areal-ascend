"""Forge — Monarch-native declarative actor framework for distributed RL training.

Forge is a framework-agnostic orchestration layer built on Monarch actors.
Training/inference/reward backends are pluggable via protocols defined in
``forge.core.protocols``.

Key components::

    ForgeActor          Base class for all actors (inherits monarch.Actor)
    Generator           vLLM/SGLang inference actor
    TrainerActor        Training actor (backend-agnostic)
    RewardActor         Reward computation actor (backend-agnostic)
    ReplayBuffer        Distributed replay buffer with eviction
    AgentActor          Multi-turn agent with tool calling
    SandboxActor        Code execution sandbox

Quick start::

    from forge.actors.base import ForgeActor
    from forge.actors.generator import Generator
    from forge.actors.trainer import TrainerActor
    from forge.core.protocols import TrainBackend, RewardBackend
    from forge.provisioner import init_provisioner, shutdown
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular import deadlocks with Monarch deserialization."""
    _lazy_map = {
        "ForgeActor": "forge.actors.base",
        "ProcessConfig": "forge.types",
        "ServiceConfig": "forge.types",
        "TrainBatch": "forge.types",
        "TrainBackend": "forge.core.protocols",
        "RewardBackend": "forge.core.protocols",
        "InferenceBridge": "forge.core.protocols",
        "ForgeConfig": "forge.core.config",
        "init_provisioner": "forge.provisioner",
        "shutdown": "forge.provisioner",
    }
    if name in _lazy_map:
        import importlib

        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'forge' has no attribute {name!r}")


__all__ = [
    "ForgeActor",
    "ProcessConfig",
    "ServiceConfig",
    "TrainBatch",
    "TrainBackend",
    "RewardBackend",
    "InferenceBridge",
    "ForgeConfig",
    "init_provisioner",
    "shutdown",
]
