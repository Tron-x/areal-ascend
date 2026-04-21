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

        # NB: torchstore's staging pool is REQUIRED on this worker.  It
        # provides 2 MiB-aligned NPU buffers; disabling it (by unsetting
        # ``TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE``) and routing RDMA
        # straight into vLLM's ``param.data`` fails with
        # ``hixl_transfer_write ret=503900`` on the first small
        # parameter (rotary / norm bias / etc.) that lands at a
        # non-2-MiB-aligned NPU offset, even in RoCE mode.  The
        # pool->param.data copy we pay on every ``ts.get`` is the price
        # of admission for bypassing those alignment constraints.
        # See the flat-buffer discussion in this file's docstring for
        # the alternative (one big aligned transfer instead).
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

        # Phase timers -- so we can see how the 1+ seconds splits
        # between (parallel RDMA transfers) and (vLLM's in-model fused
        # concatenation).  Optimizing the right bucket matters; the
        # two live in very different layers.
        timing: dict[str, float] = {
            "direct_rdma_s": 0.0,
            "fused_rdma_s": 0.0,
            "fused_load_s": 0.0,
        }

        async def _pull_direct() -> None:
            if not direct_items:
                return
            t0 = time.perf_counter()
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
            timing["direct_rdma_s"] = time.perf_counter() - t0

        async def _pull_fused() -> tuple[list[tuple[str, torch.Tensor]], int]:
            """Fetch all fused HF tensors concurrently, return
            (name, tensor) pairs + total bytes so we can feed them
            through one bulk ``model.load_weights`` call."""
            if not fused_items:
                return [], 0
            t0 = time.perf_counter()
            tensors = await asyncio.gather(*[ts.get(it["key"]) for it in fused_items])
            timing["fused_rdma_s"] = time.perf_counter() - t0
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
                t0 = time.perf_counter()
                model.load_weights(fused_pairs)
                timing["fused_load_s"] = time.perf_counter() - t0
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
            f"({gbps:.2f} GB/s)  [direct_rdma={timing['direct_rdma_s']:.2f}s, "
            f"fused_rdma={timing['fused_rdma_s']:.2f}s, "
            f"fused_load={timing['fused_load_s']:.2f}s]"
        )
        return {
            "success": True,
            "message": f"direct={direct_count}, fused={fused_count}",
            "bytes": total_bytes,
            "num_keys": direct_count + fused_count,
            "workers_load_s": workers_load_s,
            "direct_count": direct_count,
            "fused_count": fused_count,
            "direct_rdma_s": timing["direct_rdma_s"],
            "fused_rdma_s": timing["fused_rdma_s"],
            "fused_load_s": timing["fused_load_s"],
        }

    @endpoint
    def pull_weights_flat(
        self,
        version: int,
        key: str,
        plan: list | None = None,
        total_bytes: int = 0,
        shard_ranges: list | None = None,
        shard_key_fmt: str = "",
    ) -> dict:
        """Flat-buffer pull: one ``ts.get`` of the packed state_dict, then
        local unpack into model parameters.

        Counterpart of :meth:`TrainerActor.publish_weights_flat`.  The
        trainer has packed every HF tensor into a single 2 MiB-aligned
        NPU buffer and stored it as one torchstore key.  We:

        1. Allocate a matching 2 MiB-aligned NPU flat buffer via
           ``alloc_aligned_tensor``.  Because the buffer is aligned at
           allocation time, the whole RDMA path bypasses torchstore's
           staging pool (``allocate()`` sees a pool-aligned tensor and
           skips the ``pool.alloc + staged.copy_`` detour).
        2. Single ``ts.get(key, inplace_tensor=flat)`` -- one HiXL
           handshake, one TransferSync of the whole state_dict.  This is
           the same code path our standalone harness
           (``forge/scripts/test_weight_sync_2node.py``) uses to measure
           ~17 GB/s steady-state on the same hardware.
        3. Walk ``plan`` to classify each HF name as *direct* (present
           in ``model.named_parameters()``) or *fused* (needs
           ``model.load_weights`` to dispatch to vLLM's fused slot):
           direct -> ``param.data.copy_(view)`` (NPU->NPU); fused ->
           accumulate ``(name, view)`` pairs for one bulk
           ``model.load_weights(fused_pairs)`` at the end.

        The view-over-flat tensors stay alive until ``load_weights``
        returns because we keep a reference to ``flat`` in
        ``self._pulled_flat`` (same trick the trainer side uses to
        avoid tearing down HiXL registration mid-pull).
        """
        if plan is None or total_bytes <= 0:
            return {
                "success": False,
                "message": (
                    "pull_weights_flat: payload needs 'plan' + "
                    "'total_bytes'.  Check the trainer published via "
                    "publish_weights_flat."
                ),
                "bytes": 0,
                "num_keys": 0,
            }

        import asyncio
        import time

        import torch
        import torchstore as ts

        from forge.engines.weight_sync._flat_layout import torch_dtype_from_str

        model = self.worker.model_runner.model
        name_to_param = dict(model.named_parameters())

        overall_t0 = time.perf_counter()
        timing: dict[str, float] = {
            "alloc_s": 0.0,
            "rdma_s": 0.0,
            "unpack_s": 0.0,
            "fused_load_s": 0.0,
        }

        # Step 1: allocate a 2 MiB-aligned NPU flat buffer matching the
        # trainer's size.  alloc_aligned_tensor is the same API the
        # trainer uses, so the destination comes out of vLLM's NPU
        # address space already aligned.
        t0 = time.perf_counter()
        # torch.accelerator fallback for the device; vLLM workers run on
        # NPU 0 in our setup.
        import torch_npu  # noqa: F401
        from monarch._src.rdma.xdma import alloc_aligned_tensor

        flat, _raw = alloc_aligned_tensor(
            (total_bytes,), dtype=torch.uint8, device="npu:0"
        )
        timing["alloc_s"] = time.perf_counter() - t0

        # Step 2: fetch the flat buffer.  Two shapes depending on how
        # the trainer published:
        #
        # * legacy / single-key: one rank writes the whole thing
        #   under ``key``; we do one ``ts.get(key, inplace=flat)``.
        # * B1.1 shard-parallel: each trainer rank writes its byte
        #   range under ``{key}.shard_{rank}``.  We dispatch one
        #   ``ts.get`` per shard concurrently, each landing a
        #   contiguous slice of ``flat`` at the matching offset.
        #   LocalRankStrategy routes get i to volume i, so all N
        #   RoCE NICs receive in parallel.
        async def _do_get_single() -> None:
            await ts.get(key, inplace_tensor=flat)

        async def _do_get_shards() -> None:
            tasks = []
            for entry in shard_ranges or []:
                # entry is (rank, start, end)
                r_rank, r_start, r_end = entry
                shard_key = shard_key_fmt.format(rank=int(r_rank))
                shard_view = flat[int(r_start) : int(r_end)]
                tasks.append(ts.get(shard_key, inplace_tensor=shard_view))
            # Fire all N shards together; torchstore's per-key
            # transport_buffer setup runs in parallel under
            # ``asyncio.gather``.
            await asyncio.gather(*tasks)

        use_shards = bool(shard_ranges) and bool(shard_key_fmt)

        t0 = time.perf_counter()
        try:
            if use_shards:
                asyncio.run(_do_get_shards())
            else:
                asyncio.run(_do_get_single())
        except Exception as e:
            logger.exception("pull_weights_flat: ts.get failed")
            return {
                "success": False,
                "message": (
                    f"pull_weights_flat {'shards' if use_shards else 'key=' + key}: {e}"
                ),
                "bytes": 0,
                "num_keys": 0,
            }
        # Wait for any DMA tails -- the RDMA is usually sync at the HiXL
        # layer but synchronize here to make the timing meaningful.
        torch.npu.synchronize()
        timing["rdma_s"] = time.perf_counter() - t0

        # Step 3: slice flat into views and dispatch direct vs fused.
        t0 = time.perf_counter()
        direct_count = 0
        fused_pairs: list[tuple[str, torch.Tensor]] = []
        for name, shape, dtype_str, offset, nbytes in plan:
            td = torch_dtype_from_str(dtype_str)
            view = flat[offset : offset + nbytes].view(td).view(tuple(shape))
            if name in name_to_param:
                name_to_param[name].data.copy_(view)
                direct_count += 1
            else:
                fused_pairs.append((name, view))
        torch.npu.synchronize()
        timing["unpack_s"] = time.perf_counter() - t0

        if fused_pairs:
            t0 = time.perf_counter()
            model.load_weights(fused_pairs)
            timing["fused_load_s"] = time.perf_counter() - t0

        # Retain the flat buffer until the next pull so HiXL's
        # registration isn't torn down while vLLM may still be touching
        # the fused-view references inside its own load path.
        self._pulled_flat = flat

        fused_count = len(fused_pairs)
        workers_load_s = time.perf_counter() - overall_t0
        gbps = total_bytes / workers_load_s / (1024**3) if workers_load_s > 0 else 0.0
        logger.info(
            f"[WorkerWrapper] pull_weights_flat v{version}: "
            f"{direct_count} direct + {fused_count} fused, "
            f"{total_bytes / 1024**3:.2f} GB, {workers_load_s:.2f}s "
            f"({gbps:.2f} GB/s)  "
            f"[alloc={timing['alloc_s']:.2f}s, "
            f"rdma={timing['rdma_s']:.2f}s, "
            f"unpack={timing['unpack_s']:.2f}s, "
            f"fused_load={timing['fused_load_s']:.2f}s]"
        )
        return {
            "success": True,
            "message": f"flat direct={direct_count} fused={fused_count}",
            "bytes": total_bytes,
            "num_keys": direct_count + fused_count,
            "workers_load_s": workers_load_s,
            "direct_count": direct_count,
            "fused_count": fused_count,
            **{f"{k}": v for k, v in timing.items()},
        }

    @endpoint
    def destroy_process_group(self) -> None:
        """Destroy PyTorch distributed process group for clean shutdown."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
