"""TorchTitan training backend for Forge.

Uses Meta's TorchTitan (via ``ForgeEngine``) as the training engine.
TorchTitan handles all distributed training details (FSDP2, TP, PP, CP, EP,
mixed precision, checkpoint) so this adapter is thin.

Supports: Qwen3 (0.6B - 235B), Llama3/4, DeepSeek-V3, and any model
registered in TorchTitan's train_spec registry.

Usage::

    engine = create_engine(backend="titan", config={
        "model_name": "qwen3",
        "model_flavor": "1.7B",
        "max_steps": 100,
    })
"""

from forge.engines.titan.adapter import TitanTrainEngine

__all__ = ["TitanTrainEngine"]
