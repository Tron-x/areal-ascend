"""AReaL adapter — plugs AReaL's PPOTrainer/FSDPEngine into Forge actors.

All ``areal.*`` imports are confined to this package. Forge core and actors
never import from ``areal`` directly.
"""

from forge.adapters.areal.config_bridge import AReaLConfigBridge
from forge.adapters.areal.reward_backend import AReaLRewardBackend
from forge.adapters.areal.train_backend import AReaLTrainBackend

__all__ = [
    "AReaLConfigBridge",
    "AReaLRewardBackend",
    "AReaLTrainBackend",
]
