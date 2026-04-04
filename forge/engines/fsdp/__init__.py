"""FSDP2-based training engine for Forge.

Provides a native PyTorch FSDP2 training engine that implements the
``TrainEngine`` protocol without depending on AReaL's PPOTrainer.

Components:
    - ``FSDPTrainEngine``: Full ``TrainEngine`` implementation.
    - ``FSDPBatchAdapter``: ``BatchAdapter`` for FSDP2 tensor layouts.
"""

from forge.engines.fsdp.train_engine import FSDPTrainEngine

__all__ = ["FSDPTrainEngine"]
