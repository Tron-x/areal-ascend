"""Compatibility shim — re-exports from ``forge.adapters.areal.weight_sync``."""

from forge.engines.areal.weight_sync import (
    build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime,
    monarch_xccl_weight_update_alloc_mode_flat,
    optional_flat_inference_xccl_from_env,
    resolve_xccl_alloc_mode,
)

__all__ = [
    "build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime",
    "monarch_xccl_weight_update_alloc_mode_flat",
    "optional_flat_inference_xccl_from_env",
    "resolve_xccl_alloc_mode",
]
