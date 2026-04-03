"""Global resource provisioner for Monarch ProcMesh lifecycle management.

Ported from TorchForge's provisioner with Ascend NPU support via DeviceProxy.
Manages GPU allocation, ProcMesh creation, environment setup, and shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid

import torch
from monarch.actor import Actor, ProcMesh, endpoint, this_host

try:
    from monarch.utils import setup_env_for_distributed
except ImportError:
    try:
        from monarch.spmd import (
            setup_torch_elastic_env_async as setup_env_for_distributed,
        )
    except ImportError:
        setup_env_for_distributed = None

from areal.monarch_plugin.types import ProcessConfig, ProvisionerConfig

logger = logging.getLogger(__name__)


class DeviceProxy:
    """Hardware-agnostic accelerator proxy using torch.accelerator.

    Handles device counting and environment variable mapping for
    CUDA, Ascend NPU, and Intel XPU backends.
    """

    _VISIBLE_DEVICES_ENV_MAP: dict[str, str] = {
        "cuda": "CUDA_VISIBLE_DEVICES",
        "xpu": "ZE_AFFINITY_MASK",
        "npu": "ASCEND_RT_VISIBLE_DEVICES",
    }

    @staticmethod
    def is_available() -> bool:
        return torch.accelerator.is_available()

    @staticmethod
    def get_device_count() -> int:
        if not DeviceProxy.is_available():
            return 0
        return torch.accelerator.device_count()

    @classmethod
    def get_visible_devices_env_var(cls) -> str | None:
        if not cls.is_available():
            return None
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            return None
        return cls._VISIBLE_DEVICES_ENV_MAP.get(accelerator.type)

    @classmethod
    def get_isolation_env_vars(cls, device_ids: list[str]) -> dict[str, str]:
        env_var_name = cls.get_visible_devices_env_var()
        if env_var_name is None:
            return {}
        return {env_var_name: ",".join(device_ids)}

    @classmethod
    def get_visible_devices_from_env(cls) -> set[int] | None:
        env_var = cls.get_visible_devices_env_var()
        if env_var is None:
            return None
        env_value = os.environ.get(env_var, None)
        if env_value is None or not env_value.strip():
            return None
        try:
            return set(int(x.strip()) for x in env_value.split(",") if x.strip())
        except ValueError as e:
            raise ValueError(
                f"Invalid {env_var} format: '{env_value}'. "
                f"Expected comma-separated integers. Error: {e}"
            ) from e


def _get_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        return str(port)


class _RemoteInfoFetcher(Actor):
    """Temporary actor to fetch hostname and port from a remote host."""

    @endpoint
    def get_info(self) -> tuple[str, str]:
        return socket.gethostname(), _get_port()

    @endpoint
    def get_gpu_count(self) -> int:
        return DeviceProxy.get_device_count()


class EnvSetter(Actor):
    """Actor to set environment variables on each proc in a ProcMesh.

    Replaces bootstrap-based env propagation to avoid Monarch's
    SetupActor shutdown issues.
    """

    @endpoint
    def set_env(self, env_vars: dict[str, str]):
        for k, v in env_vars.items():
            os.environ[k] = v


async def get_remote_info(host_mesh) -> tuple[str, str]:
    """Get hostname and port from a host mesh."""
    throwaway_procs = host_mesh.spawn_procs(per_host={"procs": 1})
    fetcher = throwaway_procs.spawn("_fetcher", _RemoteInfoFetcher)
    singleton_slice = {k: slice(0, 1) for k in fetcher.extent.keys()}
    fetcher = fetcher.slice(**singleton_slice)
    host, port = await fetcher.get_info.call_one()
    return host, port


async def get_host_gpus(host_mesh) -> int:
    """Get the number of accelerator devices on a host mesh."""
    throwaway_procs = host_mesh.spawn_procs(per_host={"procs": 1})
    fetcher = throwaway_procs.spawn("_gpu_counter", _RemoteInfoFetcher)
    singleton_slice = {k: slice(0, 1) for k in fetcher.extent.keys()}
    fetcher = fetcher.slice(**singleton_slice)
    return await fetcher.get_gpu_count.call_one()


async def set_environment(proc_mesh: ProcMesh, env_vars: dict[str, str]):
    """Set environment variables on all procs in a mesh via EnvSetter actor."""
    env_setter = proc_mesh.spawn("_env_setter", EnvSetter)
    await env_setter.set_env.call(env_vars)


class GpuManager:
    """Tracks and assigns accelerator device IDs on a single host."""

    def __init__(
        self,
        available_devices: set[int] | None = None,
        max_device_count: int = 8,
    ):
        if available_devices is None:
            available_devices = set(range(max_device_count))
        else:
            if available_devices:
                max_device_count = max(max(available_devices) + 1, max_device_count)
        self.available_gpus = available_devices
        self.max_device_count = max_device_count

    def get_available_gpus(self) -> list[str]:
        return [str(gpu) for gpu in sorted(self.available_gpus)]

    def get_gpus(self, num_gpus: int) -> list[str]:
        if num_gpus > len(self.available_gpus):
            raise RuntimeError(
                f"Not enough GPUs available: requested {num_gpus}, "
                f"have {len(self.available_gpus)}"
            )
        gpus = sorted(self.available_gpus)[:num_gpus]
        self.available_gpus -= set(gpus)
        return [str(gpu) for gpu in gpus]

    def release_gpus(self, gpu_ids: list[str]) -> None:
        for gpu_id in gpu_ids:
            self.available_gpus.add(int(gpu_id))


class Provisioner:
    """Global resource provisioner managing ProcMesh lifecycle and GPU allocation."""

    def __init__(self, cfg: ProvisionerConfig | None = None):
        self._lock = asyncio.Lock()
        self._this_host_id = uuid.uuid1()

        available_local_devices = DeviceProxy.get_visible_devices_from_env()
        local_device_count = DeviceProxy.get_device_count()

        self._host_gpu_map: dict[uuid.UUID, GpuManager] = {
            self._this_host_id: GpuManager(
                available_local_devices, max_device_count=local_device_count
            ),
        }
        self._proc_host_map: dict[ProcMesh, object] = {}
        self._registered_actors: list = []
        self._registered_services: list = []
        self._cfg = cfg

    async def initialize(self):
        """Post-construction async initialization."""
        pass

    async def get_proc_mesh(
        self,
        num_procs: int,
        mesh_name: str | None = None,
        with_gpus: bool = False,
        num_hosts: int | None = None,
        host_mesh=None,
        env_vars: dict[str, str] | None = None,
        addr: str | None = None,
        port: str | None = None,
    ) -> ProcMesh:
        """Allocate a ProcMesh with optional GPU isolation.

        Args:
            num_procs: Number of processes.
            mesh_name: Name for the ProcMesh.
            with_gpus: Whether to allocate accelerator devices.
            num_hosts: Number of hosts (None = local).
            host_mesh: Pre-existing host mesh to use.
            env_vars: Additional environment variables.
            addr: Distributed address override.
            port: Distributed port override.

        Returns:
            A configured ProcMesh.
        """
        if env_vars is None:
            env_vars = {}

        is_remote = num_hosts is not None and num_hosts > 0

        async with self._lock:
            if is_remote and host_mesh is not None:
                host_id = getattr(host_mesh, "_host_id", uuid.uuid1())
                if host_id not in self._host_gpu_map:
                    remote_gpu_count = await get_host_gpus(host_mesh)
                    self._host_gpu_map[host_id] = GpuManager(
                        max_device_count=remote_gpu_count
                    )
                    host_mesh._host_id = host_id
                gpu_manager = self._host_gpu_map[host_id]
            else:
                host_mesh = this_host()
                gpu_manager = self._host_gpu_map[self._this_host_id]
                host_mesh._host_id = self._this_host_id

            if with_gpus:
                if not addr or not port:
                    addr, port = await get_remote_info(host_mesh)
                gpu_ids = gpu_manager.get_gpus(num_procs)

                env_vars["MASTER_ADDR"] = addr
                env_vars["MASTER_PORT"] = port

                world_size = num_procs * (num_hosts or 1)
                env_vars["WORLD_SIZE"] = str(world_size)

                env_vars.update(DeviceProxy.get_isolation_env_vars(gpu_ids))

                _propagate_ascend_env(env_vars)

            procs = host_mesh.spawn_procs(
                per_host={"procs": num_procs},
                name=mesh_name,
            )

            if env_vars:
                await set_environment(procs, env_vars)

            if with_gpus and setup_env_for_distributed is not None:
                await setup_env_for_distributed(
                    procs,
                    master_addr=addr,
                    master_port=int(port),
                )

            if with_gpus:
                procs._gpu_ids = gpu_ids

            procs._host = host_mesh
            self._proc_host_map[procs] = host_mesh

        return procs

    async def allocate_gpu_ids(
        self,
        host_mesh,
        num_gpus: int,
    ) -> list[str]:
        """Allocate GPU IDs without spawning processes.

        Args:
            host_mesh: The host mesh to allocate GPUs on.
            num_gpus: Number of GPUs to allocate.

        Returns:
            List of allocated GPU IDs as strings.
        """
        async with self._lock:
            host_id = getattr(host_mesh, "_host_id", None) or self._this_host_id
            gpu_manager = self._host_gpu_map.get(host_id)
            if gpu_manager is None:
                raise RuntimeError(f"No GPU manager found for host {host_id}")
            return gpu_manager.get_gpus(num_gpus)

    async def stop_proc_mesh(self, proc_mesh: ProcMesh):
        """Stop a ProcMesh and release its GPU resources."""
        if proc_mesh not in self._proc_host_map:
            logger.warning("ProcMesh not registered with provisioner, skipping stop.")
            return
        async with self._lock:
            if hasattr(proc_mesh, "_gpu_ids"):
                host = self._proc_host_map[proc_mesh]
                host_id = getattr(host, "_host_id", self._this_host_id)
                gpu_manager = self._host_gpu_map.get(host_id)
                if gpu_manager:
                    gpu_manager.release_gpus(proc_mesh._gpu_ids)
            await proc_mesh.stop()
            del self._proc_host_map[proc_mesh]

    def register_service(self, service) -> None:
        self._registered_services.append(service)

    def register_actor(self, actor) -> None:
        self._registered_actors.append(actor)

    async def shutdown_all_allocations(self):
        """Gracefully shut down all tracked actors and services."""
        logger.info(
            f"Shutting down {len(self._registered_services)} service(s) "
            f"and {len(self._registered_actors)} actor(s)..."
        )
        for service in reversed(self._registered_services):
            try:
                await service.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shut down service: {e}")

        for actor in reversed(self._registered_actors):
            try:
                actor_cls = getattr(actor, "_class", None) or actor.__class__
                if hasattr(actor_cls, "shutdown"):
                    await actor_cls.shutdown(actor)
            except Exception as e:
                logger.warning(f"Failed to shut down actor: {e}")

        self._registered_actors.clear()
        self._registered_services.clear()

    async def shutdown(self):
        """Tear down all remaining allocations."""
        await self.shutdown_all_allocations()
        try:
            from monarch.actor import shutdown_context

            await shutdown_context()
        except Exception as e:
            logger.warning(f"Failed to shutdown Monarch context: {e}")


def _propagate_ascend_env(env_vars: dict[str, str]) -> None:
    """Propagate Ascend-specific environment variables if present."""
    ascend_keys = [
        "ASCEND_CUSTOM_OPP_PATH",
        "LD_LIBRARY_PATH",
        "ASCEND_HOME_PATH",
    ]
    for key in ascend_keys:
        val = os.environ.get(key)
        if val and key not in env_vars:
            env_vars[key] = val


_provisioner: Provisioner | None = None


async def init_provisioner(cfg: ProvisionerConfig | None = None) -> Provisioner:
    """Initialize the global singleton provisioner."""
    global _provisioner
    if not _provisioner:
        _provisioner = Provisioner(cfg)
        await _provisioner.initialize()
    return _provisioner


async def _get_provisioner() -> Provisioner:
    if not _provisioner:
        await init_provisioner()
    return _provisioner


async def get_proc_mesh(
    process_config: ProcessConfig,
    host_mesh=None,
    env_vars: dict[str, str] | None = None,
    port: str | None = None,
    addr: str | None = None,
) -> ProcMesh:
    """Module-level convenience: allocate a ProcMesh from the global provisioner."""
    provisioner = await _get_provisioner()
    return await provisioner.get_proc_mesh(
        num_procs=process_config.procs,
        with_gpus=process_config.with_gpus,
        num_hosts=process_config.hosts,
        mesh_name=process_config.mesh_name,
        host_mesh=host_mesh,
        env_vars=env_vars,
        port=port,
        addr=addr,
    )


async def register_service(service) -> None:
    provisioner = await _get_provisioner()
    provisioner.register_service(service)


async def register_actor(actor) -> None:
    provisioner = await _get_provisioner()
    provisioner.register_actor(actor)


async def stop_proc_mesh(proc_mesh: ProcMesh):
    provisioner = await _get_provisioner()
    return await provisioner.stop_proc_mesh(proc_mesh=proc_mesh)


async def shutdown():
    """Shut down the global provisioner and all managed resources."""
    logger.info("Shutting down provisioner...")
    provisioner = await _get_provisioner()
    await provisioner.shutdown()
    logger.info("Shutdown completed successfully")
