"""Backward-compat shim package.

Real implementations live in :mod:`areal.weight_sync.vllm_ext`.  Existing
imports such as ``from areal.engine.vllm_ext.vllm_worker_extension import
VLLMWorkerExtension`` are preserved by the per-module shims in this
package.
"""
