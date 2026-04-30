"""Weight synchronization helpers shared across AReaL and Forge.

This package extracts AReaL's collective-style weight sync primitives
(previously in ``areal.engine.core.distributed`` and
``areal.engine.vllm_ext``) into a standalone, reusable module.  It is the
foundation that:

* AReaL's own ``FSDPEngine``/``MegatronEngine`` use to broadcast trainer
  weights into rollout engines.
* Forge's ``areal_xccl`` weight-sync backend wraps under its
  ``WeightSyncBackend`` Protocol so the same code path is available to
  Monarch-managed jobs.
* Third-party trainers (ms-swift, TRL, verl) can adopt to get the
  multi-host NPU-friendly path that bypasses vllm-ascend's single-host
  ``HcclCommInitRootInfo``.

The collective backend (``init_custom_process_group`` + the vLLM
``WorkerExtension`` running ``torch.distributed.broadcast`` on a secondary
ProcessGroup) is what we expose first.  A future "one-sided" backend
(TorchStore, awex, RDMA) will be added alongside as a sibling subpackage
without changing this one.

Public API:

    # Trainer-side primitives
    from areal.weight_sync import init_custom_process_group
    from areal.weight_sync import WeightSyncClient

    # vLLM server-side primitives
    from areal.weight_sync.vllm_ext import VLLMWorkerExtension
    from areal.weight_sync.vllm_ext import server_router  # FastAPI router

The legacy import paths
``areal.engine.core.distributed.init_custom_process_group`` and
``areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension`` are
preserved as thin re-export shims so existing deployments keep working.
"""

from areal.weight_sync.distributed import (
    init_custom_process_group,
    patch_dist_group_timeout,
)
from areal.weight_sync.vllm_ext.client import (
    WeightSyncClient,
    WeightSyncError,
    iter_named_tensors_for_broadcast,
)

__all__ = [
    "WeightSyncClient",
    "WeightSyncError",
    "init_custom_process_group",
    "iter_named_tensors_for_broadcast",
    "patch_dist_group_timeout",
]
