"""Forge actors — declarative Monarch actor implementations.

Import actors directly from their modules to avoid circular import
deadlocks during Monarch actor deserialization::

    from forge.actors.base import ForgeActor
    from forge.actors.generator import Generator
    from forge.actors.trainer import TrainerActor
"""


def __getattr__(name: str):
    """Lazy imports to avoid deadlocks when Monarch deserializes actors."""
    _lazy_map = {
        "ForgeActor": "forge.actors.base",
        "AgentActor": "forge.actors.agent",
        "Generator": "forge.actors.generator",
        "GroupBuffer": "forge.actors.group_buffer",
        "ReplayBuffer": "forge.actors.replay_buffer",
        "RewardActor": "forge.actors.reward",
        "SandboxActor": "forge.actors.sandbox",
        "TrainerActor": "forge.actors.trainer",
        "WeightSyncActor": "forge.actors.weight_sync",
        "WeightSyncActorHandle": "forge.actors.weight_sync",
        "spawn_weight_sync_actor": "forge.actors.weight_sync",
    }
    if name in _lazy_map:
        import importlib

        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'forge.actors' has no attribute {name!r}")


__all__ = [
    "ForgeActor",
    "AgentActor",
    "Generator",
    "GroupBuffer",
    "ReplayBuffer",
    "RewardActor",
    "SandboxActor",
    "TrainerActor",
    "WeightSyncActor",
    "WeightSyncActorHandle",
    "spawn_weight_sync_actor",
]
