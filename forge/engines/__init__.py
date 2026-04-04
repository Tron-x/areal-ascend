"""Pluggable engine backends for Forge.

Each sub-package implements the ``TrainEngine`` / ``InferenceEngine`` /
``RewardFn`` protocols from ``forge.core.protocols`` for a specific
training framework.

Available engines:
    - ``areal``: AReaL (PPOTrainer + FSDPEngine + vLLM) — legacy ``TrainBackend``
    - ``fsdp``: Native PyTorch FSDP2 — new ``TrainEngine`` protocol
    - (future) ``megatron``: Megatron-LM
    - (future) ``slime``: Slime (Megatron + SGLang)

Use ``create_engine()`` to instantiate an engine by name.
"""

from __future__ import annotations

from typing import Any


def create_engine(backend: str = "areal", **kwargs: Any):
    """Factory function to create a training engine by name.

    Args:
        backend: Engine backend name (``"areal"``, ``"fsdp"``, etc.).
        **kwargs: Backend-specific configuration.

    Returns:
        ``TrainBackend`` for ``"areal"`` (legacy), ``TrainEngine`` for others.

    Raises:
        ValueError: If the backend is not recognized.
    """
    if backend == "areal":
        from forge.engines.areal import AReaLTrainBackend

        return AReaLTrainBackend(**kwargs)
    if backend == "fsdp":
        from forge.engines.fsdp.train_engine import FSDPTrainEngine

        return FSDPTrainEngine(**kwargs)
    raise ValueError(
        f"Unknown engine backend: {backend!r}. "
        f"Available: 'areal', 'fsdp'. "
        f"Contribute new engines in forge/engines/<name>/."
    )


def create_batch_adapter(backend: str = "areal", **kwargs):
    """Factory function to create a batch adapter by name.

    Args:
        backend: Engine backend name.
        **kwargs: Backend-specific options (e.g. ``pad_token_id``).

    Returns:
        An object implementing ``BatchAdapter``.
    """
    if backend == "areal":
        from forge.engines.areal.batch_adapter import AReaLBatchAdapter

        return AReaLBatchAdapter(**kwargs)
    if backend == "fsdp":
        from forge.engines.fsdp.batch_adapter import FSDPBatchAdapter

        return FSDPBatchAdapter(**kwargs)
    raise ValueError(
        f"Unknown batch adapter: {backend!r}. "
        f"Available: 'areal', 'fsdp'. "
        f"Contribute new adapters in forge/engines/<name>/batch_adapter.py."
    )


def create_config_bridge(backend: str = "areal"):
    """Factory function to create a config bridge by name.

    Args:
        backend: Engine backend name.

    Returns:
        A config bridge for the specified backend.
    """
    if backend == "areal":
        from forge.engines.areal import AReaLConfigBridge

        return AReaLConfigBridge()
    raise ValueError(f"Unknown config bridge: {backend!r}")
