"""WorkerWrapper and WorkerRegistry for Monarch-distributed vLLM workers.

WorkerWrapper extends vLLM's WorkerWrapperBase with Monarch Actor endpoints.
WorkerRegistry bridges the EngineCore subprocess / Generator actor boundary.
"""

from __future__ import annotations

import logging
from typing import Any

from monarch.actor import Actor, context, endpoint
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = logging.getLogger(__name__)


class WorkerRegistry(Actor):
    """Rendezvous point for cross-process worker registration.

    Spawned by Generator on a CPU proc. MonarchExecutor (inside EngineCore
    subprocess) registers workers here; Generator queries after init.
    """

    def __init__(self):
        self._workers = None

    @endpoint
    def register_workers(self, workers_mesh) -> None:
        self._workers = workers_mesh
        logger.info(f"[WorkerRegistry] Workers registered: {workers_mesh}")

    @endpoint
    def get_workers(self):
        return self._workers


class _FutureWrapper:
    """Adapts Monarch Future to vLLM's concurrent.futures.Future interface."""

    def __init__(self, monarch_future, timeout):
        self._future = monarch_future
        self._timeout = timeout
        self._result = None

    def result(self, timeout=None):
        if self._result is None:
            use_timeout = timeout if timeout is not None else self._timeout
            try:
                result = self._future.get(timeout=use_timeout)
            except TimeoutError as e:
                raise TimeoutError(
                    f"Monarch RPC timed out after {use_timeout}s."
                ) from e
            except Exception as e:
                raise RuntimeError(f"Monarch RPC failed: {e}") from e
            outputs = [value for _, value in result.items()]
            self._result = outputs
        return self._result[0] if self._result else None

    def __getitem__(self, index):
        if index == 0:
            return self
        raise IndexError(f"FutureWrapper only supports index 0, got {index}")


class WorkerWrapper(WorkerWrapperBase, Actor):
    """Monarch actor wrapper around vLLM WorkerWrapperBase.

    Inherits all vLLM worker lifecycle methods and exposes them via
    the ``execute_method`` endpoint for dynamic dispatch.
    """

    def __init__(self, vllm_config):
        rank = context().actor_instance.rank.rank
        WorkerWrapperBase.__init__(self, rpc_rank=rank, global_rank=rank)
        Actor.__init__(self)
        self._vllm_config = vllm_config

    def init_worker(self, all_kwargs):
        monarch_rank = self.rpc_rank
        expected_rank = all_kwargs[monarch_rank].get("rank")
        assert monarch_rank == expected_rank, (
            f"Rank mismatch: Monarch={monarch_rank}, expected={expected_rank}"
        )
        super().init_worker(all_kwargs)

    @endpoint
    def execute_method(self, method: str, *args, **kwargs):
        from vllm.v1.outputs import AsyncModelRunnerOutput

        fn = getattr(self, method)
        result = fn(*args, **kwargs)
        if isinstance(result, AsyncModelRunnerOutput):
            result = result.get_output()
        return result

    @endpoint
    def update_weights(
        self,
        state_dict: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> int:
        """Legacy path: load weights from a state dict into the model.

        Kept for the non-torchstore backends and for local debugging where
        a caller already has CPU / NPU tensors in hand.  The fast path for
        ``torchstore_multi_vol`` is ``pull_weights`` below -- it runs
        inside this worker proc and lets HiXL RDMA write straight into the
        model's NPU parameters without any CPU staging.

        Args:
            state_dict: HF-format state dict to load.
            version: Policy version (for logging).

        Returns:
            Number of parameters loaded.
        """
        if state_dict is None:
            return 0
        import torch

        model = self.worker.model_runner.model
        loaded = 0
        for name, param in state_dict.items():
            device = torch.accelerator.current_accelerator()
            model.load_weights([(name, param.to(device))])
            loaded += 1
        logger.info(f"[WorkerWrapper] Loaded {loaded} weights (v{version})")
        return loaded

    @endpoint
    def pull_weights(
        self,
        version: int,
        items: list[dict] | None = None,
    ) -> dict:
        """Zero-CPU fast path: pull weights from torchstore straight into
        this worker's NPU parameter memory via HiXL RDMA.

        Each ``items[i]`` is ``{"name": str, "key": str}`` (plus optional
        ``slice_spec`` for Phase-2 TP>1).  For each item:

        1. Look up ``name`` in ``model.named_parameters()``.  vLLM's Qwen3
           (and many other models) fuses some weights (e.g. ``gate_proj ||
           up_proj`` -> ``gate_up_proj``), so HF names may not map 1:1 to
           ``named_parameters`` -- those fall back to the classic
           ``model.load_weights([(name, tensor)])`` path so vLLM's loader
           handles the fusion.
        2. For direct-map params: ``ts.get(key, inplace_tensor=param.data)``.
           torchstore + MonarchRDMA + HiXL RoCE writes the remote tensor's
           bytes straight into the NPU memory backing ``param.data``.  No
           ``.cpu()``, no ``.to(device)``, no intermediate tensor.
        3. For fused params: first ``ts.get(key)`` into a regular NPU
           tensor, then ``model.load_weights([(hf_name, tensor)])`` so
           vLLM's loader does the fused-slot assignment.

        Returns aggregate stats; the caller sums across workers if needed.
        """
        if not items:
            return {
                "success": False,
                "message": "pull_weights: empty items",
                "bytes": 0,
                "num_keys": 0,
            }

        import asyncio
        import time

        import torch
        import torchstore as ts

        model = self.worker.model_runner.model
        name_to_param = dict(model.named_parameters())

        load_t0 = time.perf_counter()
        direct_count = 0
        fused_count = 0
        total_bytes = 0

        async def _do_pulls() -> None:
            nonlocal direct_count, fused_count, total_bytes
            for item in items:
                name = item["name"]
                key = item["key"]
                param = name_to_param.get(name)
                if param is not None:
                    # Direct HiXL RDMA write into NPU param memory.
                    # Important: .data keeps the exact storage; ts.get
                    # inplace_tensor writes into the provided tensor's
                    # storage without reallocating.
                    await ts.get(key, inplace_tensor=param.data)
                    direct_count += 1
                    total_bytes += param.numel() * param.element_size()
                else:
                    # Fused-param fallback: let vLLM's loader figure out
                    # which slot (gate / up / etc.) this HF name maps to.
                    tensor = await ts.get(key)
                    if tensor is None:
                        raise RuntimeError(
                            f"pull_weights: torchstore miss for key {key!r}"
                        )
                    device = torch.accelerator.current_accelerator()
                    model.load_weights([(name, tensor.to(device))])
                    fused_count += 1
                    total_bytes += tensor.numel() * tensor.element_size()

        try:
            asyncio.run(_do_pulls())
        except Exception as e:
            logger.exception("pull_weights failed")
            return {
                "success": False,
                "message": f"pull_weights failed: {e}",
                "bytes": total_bytes,
                "num_keys": direct_count + fused_count,
            }

        workers_load_s = time.perf_counter() - load_t0
        logger.info(
            f"[WorkerWrapper] pull_weights v{version}: "
            f"{direct_count} direct + {fused_count} fused, "
            f"{total_bytes / 1024**3:.2f} GB, {workers_load_s:.2f}s "
            f"({total_bytes / workers_load_s / 1024**3:.2f} GB/s)"
            if workers_load_s > 0
            else f"[WorkerWrapper] pull_weights v{version}: "
            f"{direct_count} direct + {fused_count} fused"
        )
        return {
            "success": True,
            "message": f"direct={direct_count}, fused={fused_count}",
            "bytes": total_bytes,
            "num_keys": direct_count + fused_count,
            "workers_load_s": workers_load_s,
            "direct_count": direct_count,
            "fused_count": fused_count,
        }

    @endpoint
    def destroy_process_group(self) -> None:
        """Destroy PyTorch distributed process group for clean shutdown."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
