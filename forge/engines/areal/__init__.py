"""AReaL adapter — plugs AReaL's PPOTrainer/FSDPEngine into Forge actors.

All ``areal.*`` imports are confined to this package. Forge core and actors
never import from ``areal`` directly.
"""

from forge.engines.areal.batch_adapter import AReaLBatchAdapter
from forge.engines.areal.config_bridge import AReaLConfigBridge
from forge.engines.areal.reward_backend import AReaLRewardBackend
from forge.engines.areal.train_backend import AReaLTrainBackend

__all__ = [
    "AReaLBatchAdapter",
    "AReaLConfigBridge",
    "AReaLRewardBackend",
    "AReaLTrainBackend",
]
