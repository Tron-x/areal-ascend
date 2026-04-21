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
        ``slice_spec`` for Phase-2 TP>1).  Two routing buckets:

        * **Direct**: ``name`` has a matching ``model.named_parameters()``
          entry (no vLLM fusion).  We issue ``ts.get(key,
          inplace_tensor=param.data)`` per item and gather them all on
          one event loop so torchstore's per-request handshakes + RDMA
          transfers pipeline through the storage volume actor
          concurrently rather than serially.
        * **Fused**: ``name`` is not in ``named_parameters`` (e.g. Qwen3
          ``gate_proj`` + ``up_proj`` get concatenated into
          ``gate_up_proj`` at vLLM load time).  We ``ts.get`` each fused
          HF tensor (also concurrently) and hand the whole list to one
          ``model.load_weights(list_of_pairs)`` call -- vLLM's loader
          then walks its internal WeightsMapper to do the fused-slot
          assignment in bulk instead of us paying the per-call overhead.

        Why gather rather than ``ts.get_batch``:
        ``MonarchRDMATransportBuffer`` inherits ``supports_batch_gets =
        False`` from the base class, so even the library's ``get_batch``
        falls through to a serial ``for request in requests`` loop.
        Driving N client-side calls concurrently via ``asyncio.gather``
        bypasses that and lets each get run its own handshake + RDMA
        through the single storage-volume actor, which accepts
        concurrent RPCs.  Teaching the transport to batch internally
        (one RDMABuffer covering multiple tensors) is a separate
        torchstore-side optimization and not in scope here.
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

        # Partition up front: direct vs. fused.  Each bucket is an
        # independent list of (item, param_or_tensor_placeholder) so we
        # can launch the two batches in parallel.
        direct_items: list[dict] = []
        fused_items: list[dict] = []
        direct_bytes = 0
        for item in items:
            name = item["name"]
            if name in name_to_param:
                direct_items.append(item)
                p = name_to_param[name]
                direct_bytes += p.numel() * p.element_size()
            else:
                fused_items.append(item)

        overall_t0 = time.perf_counter()

        async def _pull_direct() -> None:
            if not direct_items:
                return
            # All direct gets in parallel -- torchstore's
            # MonarchRDMA transport runs one handshake + one RDMA
            # per-key, and concurrent calls pipeline through the
            # storage volume actor's async dispatch.
            await asyncio.gather(
                *[
                    ts.get(it["key"], inplace_tensor=name_to_param[it["name"]].data)
                    for it in direct_items
                ]
            )

        async def _pull_fused() -> tuple[list[tuple[str, torch.Tensor]], int]:
            """Fetch all fused HF tensors concurrently, return
            (name, tensor) pairs + total bytes so we can feed them
            through one bulk ``model.load_weights`` call."""
            if not fused_items:
                return [], 0
            tensors = await asyncio.gather(*[ts.get(it["key"]) for it in fused_items])
            pairs: list[tuple[str, torch.Tensor]] = []
            nbytes = 0
            device = torch.accelerator.current_accelerator()
            for it, tensor in zip(fused_items, tensors):
                if tensor is None:
                    raise RuntimeError(
                        f"pull_weights: torchstore miss for fused key {it['key']!r}"
                    )
                # torchstore returns NPU tensors when _STORAGE_DEVICE is set;
                # ``.to(device)`` is a no-op in that case but guards the
                # configuration where storage lands on a different device.
                pairs.append((it["name"], tensor.to(device)))
                nbytes += tensor.numel() * tensor.element_size()
            return pairs, nbytes

        async def _do() -> tuple[int, int]:
            # Launch direct + fused pulls concurrently.  gather waits
            # for both before returning.
            _, fused_result = await asyncio.gather(
                _pull_direct(),
                _pull_fused(),
            )
            fused_pairs, fused_nbytes = fused_result
            # One bulk load_weights call for all fused params -- lets
            # vLLM's WeightsMapper concatenate gate_proj + up_proj etc.
            # inside its own single pass rather than us paying the
            # per-call overhead N times.
            if fused_pairs:
                model.load_weights(fused_pairs)
            return len(direct_items), fused_nbytes

        try:
            direct_count, fused_bytes = asyncio.run(_do())
        except Exception as e:
            logger.exception("pull_weights failed")
            return {
                "success": False,
                "message": f"pull_weights failed: {e}",
                "bytes": 0,
                "num_keys": 0,
            }

        fused_count = len(fused_items)
        total_bytes = direct_bytes + fused_bytes
        workers_load_s = time.perf_counter() - overall_t0
        gbps = total_bytes / workers_load_s / (1024**3) if workers_load_s > 0 else 0.0
        logger.info(
            f"[WorkerWrapper] pull_weights v{version}: "
            f"{direct_count} direct + {fused_count} fused, "
            f"{total_bytes / 1024**3:.2f} GB, {workers_load_s:.2f}s "
            f"({gbps:.2f} GB/s)"
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
