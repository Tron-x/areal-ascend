"""vLLM-side weight sync extensions.

Two pieces live here:

* :class:`VLLMWorkerExtension` -- mounted into vLLM workers via
  ``--worker_extension_cls`` so the workers expose ``init_update_weight_group``
  / ``update_weight_xccl`` / ``update_weights`` as ``collective_rpc`` methods.
* The FastAPI ``router`` from :mod:`server_router` -- exposes those
  ``collective_rpc`` calls as HTTP endpoints (``/areal_init_weights_update_group``,
  ``/areal_update_weights_xccl``, ``/areal_update_weights``, ...) so trainers
  drive the sync over HTTP.

Both pieces are imported through legacy paths
``areal.engine.vllm_ext.vllm_worker_extension`` and
``areal.engine.vllm_ext.areal_vllm_server`` for backward compatibility.
"""

from areal.weight_sync.vllm_ext.client import (
    WeightSyncClient,
    WeightSyncError,
    iter_named_tensors_for_broadcast,
)
from areal.weight_sync.vllm_ext.worker_extension import VLLMWorkerExtension

__all__ = [
    "VLLMWorkerExtension",
    "WeightSyncClient",
    "WeightSyncError",
    "iter_named_tensors_for_broadcast",
]
