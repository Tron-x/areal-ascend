"""AReaLMonarchExecutor -- Monarch distributed executor for vLLM.

Ported from TorchForge's MonarchExecutor with Ascend NPU support
via DeviceProxy for automatic device isolation env var selection.
"""

from __future__ import annotations

import base64
import logging
import os
import socket
from collections.abc import Callable
from typing import Any

import cloudpickle
from vllm.v1.executor.abstract import Executor

from forge.actors.generator.worker import (
    WorkerWrapper,
    _FutureWrapper,
)
from forge.provisioner import DeviceProxy

logger = logging.getLogger(__name__)


def _get_host_ip() -> str:
    if host_ip := os.environ.get("VLLM_HOST_IP"):
        return host_ip
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return "127.0.0.1"


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
    gpu_ids: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict]]:
    """Build per-worker environment variables and init kwargs.

    Uses DeviceProxy for device-agnostic isolation env var selection
    (CUDA_VISIBLE_DEVICES, ASCEND_RT_VISIBLE_DEVICES, etc.).
    """
    if gpu_ids:
        device_str = ",".join(gpu_ids)
    else:
        device_str = ",".join(str(i) for i in range(gpus_per_host))

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    world_size = vllm_config.parallel_config.world_size

    all_envs = []
    all_kwargs = []
    for rank in range(world_size):
        local_rank = rank % gpus_per_host
        is_driver = rank % tp_size == 0

        env_vars = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "VLLM_HOST_IP": master_addr,
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "TORCH_DISABLE_ADDR2LINE": "1",
        }

        env_vars.update(DeviceProxy.get_isolation_env_vars(device_str.split(",")))

        ascend_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        if ascend_opp:
            env_vars["ASCEND_CUSTOM_OPP_PATH"] = ascend_opp
        ld_path = os.environ.get("LD_LIBRARY_PATH")
        if ld_path:
            env_vars["LD_LIBRARY_PATH"] = ld_path

        all_envs.append(env_vars)

        worker_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": "env://",
            "is_driver_worker": is_driver,
            "shared_worker_lock": None,
        }
        all_kwargs.append(worker_kwargs)

    return all_envs, all_kwargs


class AReaLMonarchExecutor(Executor):
    """Distributed vLLM executor using Monarch actors on NPU/GPU.

    Architecture::

        Generator (CPU proc)
          +-- WorkerRegistry (CPU proc, same as Generator)
          |
          +-- EngineCore subprocess
                +-- AReaLMonarchExecutor
                      +-- ProcMesh (GPU/NPU)
                            +-- WorkerWrapper x N (tensor parallel)

    MonarchExecutor owns the proc_mesh and workers. Generator owns the
    host_mesh (resource allocation).
    """

    uses_ray: bool = False
    supports_pp: bool = False
    worker_class = WorkerWrapper

    def _init_executor(self) -> None:
        from monarch.actor import enable_transport

        try:
            enable_transport("tcp")
        except RuntimeError:
            pass

        host_mesh_str = os.environ.get("VLLM_MONARCH_HOST_MESH")
        if not host_mesh_str:
            raise RuntimeError("VLLM_MONARCH_HOST_MESH not set.")

        registry_str = os.environ.get("VLLM_MONARCH_WORKER_REGISTRY")
        if not registry_str:
            raise RuntimeError("VLLM_MONARCH_WORKER_REGISTRY not set.")

        world_size = self.vllm_config.parallel_config.world_size

        self.host_mesh = cloudpickle.loads(base64.b64decode(host_mesh_str))
        self.worker_registry = cloudpickle.loads(base64.b64decode(registry_str))

        try:
            num_hosts = self.host_mesh.extent["hosts"]
        except (KeyError, AttributeError, TypeError, ValueError):
            num_hosts = 1
        gpus_per_host = world_size // num_hosts

        logger.info(
            f"[AReaLMonarchExecutor] Creating ProcMesh: "
            f"{gpus_per_host} procs/host, world_size={world_size}"
        )
        self.proc_mesh = self.host_mesh.spawn_procs(
            per_host={"procs": gpus_per_host}, name="vllm_workers"
        )
        self.workers = self.proc_mesh.spawn(
            "vllm_workers", self.worker_class, self.vllm_config
        )

        head_ip = _get_host_ip()
        master_port = _get_free_port()

        gpu_ids_str = os.environ.get("VLLM_MONARCH_GPU_IDS")
        gpu_ids = gpu_ids_str.split(",") if gpu_ids_str else None

        all_envs, all_kwargs = _build_worker_configs(
            self.vllm_config, gpus_per_host, head_ip, master_port, gpu_ids
        )

        self.collective_rpc("update_environment_variables", args=(all_envs,))
        self.collective_rpc("init_worker", args=(all_kwargs,))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        logger.info(f"[AReaLMonarchExecutor] Initialized {world_size} workers")

        self.worker_registry.register_workers.call_one(self.workers).get()
        logger.info("[AReaLMonarchExecutor] Workers registered")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any]:
        future = self.workers.execute_method.call(method, *args, **(kwargs or {}))
        if non_block:
            return _FutureWrapper(future, timeout)
        result = future.get(timeout=timeout)
        return [value for _, value in result.items()]

    def check_health(self):
        return

    def shutdown(self):
        logger.info("[AReaLMonarchExecutor] Shutting down...")
        try:
            if hasattr(self, "workers") and self.workers is not None:
                self.workers.destroy_process_group.call().get()
        except Exception as e:
            logger.warning(f"Error destroying process groups: {e}")

        try:
            if hasattr(self, "workers"):
                super().shutdown()
        except Exception as e:
            logger.warning(f"Error during worker shutdown: {e}")

        try:
            if hasattr(self, "proc_mesh") and self.proc_mesh is not None:
                self.proc_mesh.stop().get()
        except Exception as e:
            logger.warning(f"Error stopping proc_mesh: {e}")

        logger.info("[AReaLMonarchExecutor] Shutdown complete")
