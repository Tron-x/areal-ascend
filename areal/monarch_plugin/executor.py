"""Monarch distributed executor for vLLM on Ascend NPU.

Adapted from TorchForge's MonarchExecutor for AReaL integration.
Uses Monarch ProcMesh to manage vLLM workers instead of vLLM's native
subprocess management.  Preserves AReaL's VLLMWorkerExtension for
xccl weight synchronisation.

Key differences from TorchForge's ForgeMonarchExecutor:
  - No TorchStore integration (xccl weight sync is kept)
  - Worker loads VLLMWorkerExtension via vLLM's worker_extension_cls config
  - NPU device binding via ASCEND_RT_VISIBLE_DEVICES
"""

from __future__ import annotations

import base64
import logging
import os
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
from monarch.actor import Actor, context, enable_transport, endpoint
from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_host_ip() -> str:
    if host_ip := os.environ.get("VLLM_HOST_IP"):
        return host_ip
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _build_worker_configs(
    vllm_config,
    gpus_per_host: int,
    master_addr: str,
    master_port: int,
    gpu_ids: List[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict]]:
    if gpu_ids:
        device_list = ",".join(gpu_ids)
    else:
        device_list = ",".join(str(i) for i in range(gpus_per_host))

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    world_size = vllm_config.parallel_config.world_size

    all_envs: list[dict[str, str]] = []
    all_kwargs: list[dict] = []

    for rank in range(world_size):
        local_rank = rank % gpus_per_host
        is_driver = rank % tp_size == 0

        env_vars: dict[str, str] = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "VLLM_HOST_IP": master_addr,
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "TORCH_DISABLE_ADDR2LINE": "1",
        }

        from areal.infra.platforms import current_platform
        if current_platform.device_control_env_var:
            env_vars[current_platform.device_control_env_var] = device_list

        all_envs.append(env_vars)
        all_kwargs.append({
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": "env://",
            "is_driver_worker": is_driver,
            "shared_worker_lock": None,
        })

    return all_envs, all_kwargs


# ---------------------------------------------------------------------------
# WorkerRegistry -- bridge between EngineCore and GeneratorActor
# ---------------------------------------------------------------------------

class WorkerRegistry(Actor):
    """Rendezvous point so MonarchExecutor can hand the worker ActorMesh
    back to GeneratorActor."""

    def __init__(self):
        self._workers = None

    @endpoint
    def register_workers(self, workers_mesh) -> None:
        self._workers = workers_mesh
        logger.info(f"[WorkerRegistry] Workers registered: {workers_mesh}")

    @endpoint
    def get_workers(self):
        return self._workers


# ---------------------------------------------------------------------------
# _FutureWrapper -- adapts Monarch Future to vLLM's expected interface
# ---------------------------------------------------------------------------

class _FutureWrapper:
    def __init__(self, monarch_future, timeout):
        self._future = monarch_future
        self._timeout = timeout
        self._result = None

    def result(self, timeout=None):
        if self._result is None:
            t = timeout if timeout is not None else self._timeout
            result = self._future.get(timeout=t)
            self._result = [v for _, v in result.items()]
        return self._result[0] if self._result else None

    def __getitem__(self, index):
        if index == 0:
            return self
        raise IndexError(f"_FutureWrapper only supports index 0, got {index}")


# ---------------------------------------------------------------------------
# AReaLWorkerWrapper -- Monarch actor wrapping vLLM worker
# ---------------------------------------------------------------------------

class AReaLWorkerWrapper(WorkerWrapperBase, Actor):
    """vLLM worker that is also a Monarch actor.

    Exposes ``execute_method`` so that ``collective_rpc`` can invoke any
    method on the underlying worker, including VLLMWorkerExtension methods.
    """

    def __init__(self, vllm_config):
        rank = context().actor_instance.rank.rank
        WorkerWrapperBase.__init__(self, rpc_rank=rank, global_rank=rank)
        Actor.__init__(self)
        self._vllm_config_ref = vllm_config
        logger.info(f"[AReaLWorkerWrapper] Initialised rank={rank}")

    def init_worker(self, all_kwargs):
        monarch_rank = self.rpc_rank
        expected_rank = all_kwargs[monarch_rank].get("rank")
        assert monarch_rank == expected_rank, (
            f"Rank mismatch: Monarch={monarch_rank}, expected={expected_rank}"
        )
        super().init_worker(all_kwargs)
        logger.info(f"[AReaLWorkerWrapper] init_worker complete, rank={monarch_rank}")

    @endpoint
    def execute_method(self, method: str, *args, **kwargs):
        fn = getattr(self, method)
        return fn(*args, **kwargs)

    @endpoint
    def destroy_process_group(self) -> None:
        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("[AReaLWorkerWrapper] Destroying process group")
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# AReaLMonarchExecutor -- vLLM Executor using Monarch ProcMesh
# ---------------------------------------------------------------------------

_CANN_ENV_KEYS = [
    "LD_LIBRARY_PATH",
    "ASCEND_HOME_PATH",
    "ASCEND_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_AICPU_PATH",
    "ATB_HOME_PATH",
    "TOOLCHAIN_HOME",
    "PYTHONPATH",
    "PATH",
    "CMAKE_PREFIX_PATH",
]


def _capture_cann_env() -> dict:
    """Snapshot all CANN/ATB env vars from the current process."""
    return {k: os.environ[k] for k in _CANN_ENV_KEYS if k in os.environ}


def _make_worker_bootstrap(cann_env: dict, device_ids_str: str | None = None):
    """Create a bootstrap that restores CANN env vars in the worker process."""
    from areal.infra.platforms import current_platform
    device_env_var = current_platform.device_control_env_var

    def _bootstrap():
        for k, v in cann_env.items():
            os.environ[k] = v
        if device_ids_str is not None and device_env_var:
            os.environ[device_env_var] = device_ids_str

    return _bootstrap


class AReaLMonarchExecutor(Executor):
    """Distributed executor running vLLM workers on Monarch actors.

    Deserialises a HostMesh from ``VLLM_MONARCH_HOST_MESH``, creates a
    ProcMesh, spawns ``AReaLWorkerWrapper`` actors, and registers them
    with ``WorkerRegistry`` for the GeneratorActor to query.
    """

    uses_ray: bool = False
    supports_pp: bool = False
    worker_class = AReaLWorkerWrapper

    def _init_executor(self) -> None:
        try:
            enable_transport("tcp")
        except RuntimeError:
            pass

        host_mesh_str = os.environ.get("VLLM_MONARCH_HOST_MESH")
        if not host_mesh_str:
            raise RuntimeError("VLLM_MONARCH_HOST_MESH not set")

        registry_str = os.environ.get("VLLM_MONARCH_WORKER_REGISTRY")
        if not registry_str:
            raise RuntimeError("VLLM_MONARCH_WORKER_REGISTRY not set")

        world_size = self.vllm_config.parallel_config.world_size

        logger.info("[AReaLMonarchExecutor] Deserialising HostMesh ...")
        self.host_mesh = cloudpickle.loads(base64.b64decode(host_mesh_str))

        logger.info("[AReaLMonarchExecutor] Deserialising WorkerRegistry ...")
        self.worker_registry = cloudpickle.loads(base64.b64decode(registry_str))

        try:
            num_hosts = self.host_mesh.extent["hosts"]
        except (KeyError, AttributeError, TypeError, ValueError):
            num_hosts = 1
        gpus_per_host = world_size // num_hosts

        cann_env = _capture_cann_env()
        logger.info(
            f"[AReaLMonarchExecutor] Captured CANN env keys: {list(cann_env.keys())}"
        )

        gpu_ids_str = os.environ.get("VLLM_MONARCH_GPU_IDS")

        from areal.infra.platforms import current_platform
        resource_key = "npu" if current_platform.device_type == "npu" else "gpu"

        logger.info(
            f"[AReaLMonarchExecutor] Creating ProcMesh: "
            f"{gpus_per_host} {resource_key}/host, world_size={world_size}, "
            f"gpu_ids={gpu_ids_str}"
        )
        self.proc_mesh = self.host_mesh.spawn_procs(
            per_host={resource_key: gpus_per_host},
            bootstrap=_make_worker_bootstrap(cann_env, gpu_ids_str),
            name="vllm_workers",
        )
        self.workers = self.proc_mesh.spawn(
            "vllm_workers", self.worker_class, self.vllm_config
        )

        head_node_ip = _get_host_ip()
        master_port = _get_free_port()
        logger.info(f"[AReaLMonarchExecutor] Head node: {head_node_ip}:{master_port}")

        gpu_ids = gpu_ids_str.split(",") if gpu_ids_str else None

        all_envs, all_kwargs = _build_worker_configs(
            self.vllm_config, gpus_per_host, head_node_ip, master_port, gpu_ids
        )

        self.collective_rpc("update_environment_variables", args=(all_envs,))
        self.collective_rpc("init_worker", args=(all_kwargs,))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        logger.info(f"[AReaLMonarchExecutor] {world_size} workers initialised")

        self.worker_registry.register_workers.call_one(self.workers).get()
        logger.info("[AReaLMonarchExecutor] Workers registered")

    # -- Executor interface ---------------------------------------------------

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        non_block: bool = False,
    ) -> List[Any]:
        future = self.workers.execute_method.call(method, *args, **(kwargs or {}))
        if non_block:
            return _FutureWrapper(future, timeout)
        result = future.get(timeout=timeout)
        return [v for _, v in result.items()]

    def check_health(self):
        return

    def shutdown(self):
        logger.info("[AReaLMonarchExecutor] Shutting down ...")
        try:
            if hasattr(self, "workers") and self.workers is not None:
                self.workers.destroy_process_group.call().get()
                logger.info("[AReaLMonarchExecutor] Destroyed process groups")
        except Exception as e:
            logger.warning(f"[AReaLMonarchExecutor] Error destroying PGs: {e}")

        try:
            if hasattr(self, "workers"):
                super().shutdown()
        except Exception as e:
            logger.warning(f"[AReaLMonarchExecutor] Error during worker shutdown: {e}")

        try:
            if hasattr(self, "proc_mesh") and self.proc_mesh is not None:
                self.proc_mesh.stop().get()
                logger.info("[AReaLMonarchExecutor] Stopped proc_mesh")
        except Exception as e:
            logger.warning(f"[AReaLMonarchExecutor] Error stopping proc_mesh: {e}")

        logger.info("[AReaLMonarchExecutor] Shutdown complete")
