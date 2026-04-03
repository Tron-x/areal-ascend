"""Forge — Monarch-native declarative actor framework for distributed RL training.

Forge absorbs TorchForge's integrated actor model, combining it with AReaL's
agentic RL capabilities and Ascend NPU support.

Key components::

    ForgeActor          Base class for all actors (inherits monarch.Actor)
    Generator           vLLM/SGLang inference actor
    TrainerActor        FSDP/Megatron training actor
    RewardActor         Pluggable reward computation
    ReplayBuffer        Distributed replay buffer with eviction
    AgentActor          Multi-turn agent with tool calling
    SandboxActor        Code execution sandbox

Quick start::

    from forge.actors.base import ForgeActor
    from forge.actors.generator import Generator
    from forge.actors.trainer import TrainerActor
    from forge.provisioner import init_provisioner, shutdown
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular import deadlocks with Monarch deserialization."""
    _lazy_map = {
        "ForgeActor": "forge.actors.base",
        "ProcessConfig": "forge.types",
        "ServiceConfig": "forge.types",
        "TrainBatch": "forge.types",
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
    "init_provisioner",
    "shutdown",
]
