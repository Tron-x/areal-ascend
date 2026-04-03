"""Compatibility shim — re-exports from ``forge.adapters.areal.inference_bridge``.

The real implementation now lives in ``forge.adapters.areal.inference_bridge``.
This module exists only for backward compatibility with code that imports
``forge.engine.monarch_inf.MonarchVLLMEngine``.
"""

from forge.engines.areal.inference_bridge import AReaLInferenceBridge as MonarchVLLMEngine

__all__ = ["MonarchVLLMEngine"]
