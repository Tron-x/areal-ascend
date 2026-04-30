"""Backward-compat shim.

The real implementation now lives in
:mod:`areal.weight_sync.vllm_ext.worker_extension`.  This module re-exports
:class:`VLLMWorkerExtension` so existing config -- including the default
``worker_extension_cls`` value
``"areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension"``
loaded by vLLM via ``importlib`` -- keeps working unchanged.

New code should import directly from ``areal.weight_sync.vllm_ext``.
"""

from areal.weight_sync.vllm_ext.worker_extension import (
    VLLMWorkerExtension,
    _apply_ascend_patch_once,
    undo_moe_postprocess_for_reload,
)

__all__ = [
    "VLLMWorkerExtension",
    "_apply_ascend_patch_once",
    "undo_moe_postprocess_for_reload",
]
