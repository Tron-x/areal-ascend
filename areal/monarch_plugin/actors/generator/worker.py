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
        WorkerWrapperBase.__init__(self, vllm_config, rpc_rank=rank, global_rank=rank)
        Actor.__init__(self)

    def init_worker(self, all_kwargs):
        monarch_rank = self.rpc_rank
        expected_rank = all_kwargs[monarch_rank].get("rank")
        assert monarch_rank == expected_rank, (
            f"Rank mismatch: Monarch={monarch_rank}, expected={expected_rank}"
        )
        super().init_worker(all_kwargs)

    @endpoint
    def execute_method(self, method: str, *args, **kwargs):
        fn = getattr(self, method)
        return fn(*args, **kwargs)

    @endpoint
    def update_weights(
        self,
        state_dict: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> int:
        """Load weights from a state dict into the model.

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
    def destroy_process_group(self) -> None:
        """Destroy PyTorch distributed process group for clean shutdown."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
