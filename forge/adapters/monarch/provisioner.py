"""Monarch resource provisioning for Forge.

Provides:
- ``get_proc_mesh()``: create a Monarch ProcMesh with requested resources.
- ``DeviceProxy``: hardware-agnostic device management (GPU, NPU, XPU).
- Bootstrap factories for generator, trainer, and CPU-only actors.

Ported from ``areal/monarch_plugin/provisioner.py`` and
``areal/monarch_plugin/bootstraps.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger("forge.monarch.provisioner")


class DeviceProxy:
    """Hardware-agnostic device abstraction.

    Detects the available accelerator type (CUDA, NPU, XPU) and provides
    a unified interface for device count, env var names, and device setting.
    """

    def __init__(self) -> None:
        self._type = self._detect_type()

    @staticmethod
    def _detect_type() -> str:
        try:
            import torch

            if hasattr(torch, "npu") and torch.npu.is_available():
                return "npu"
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass
        return "cpu"

    @property
    def device_type(self) -> str:
        return self._type

    @property
    def device_control_env_var(self) -> str | None:
        return {
            "npu": "ASCEND_RT_VISIBLE_DEVICES",
            "cuda": "CUDA_VISIBLE_DEVICES",
            "xpu": "ZE_AFFINITY_MASK",
        }.get(self._type)

    @property
    def resource_key(self) -> str:
        """Key for Monarch ProcMesh ``per_host`` dict."""
        if self._type in ("npu", "cuda", "xpu"):
            return self._type
        return "cpu"

    def device_count(self) -> int:
        import torch

        if self._type == "npu":
            return torch.npu.device_count()
        if self._type == "cuda":
            return torch.cuda.device_count()
        if self._type == "xpu":
            return torch.xpu.device_count()
        return 0

    def set_device(self, index: int) -> None:
        import torch

        if self._type == "npu":
            torch.npu.set_device(index)
        elif self._type == "cuda":
            torch.cuda.set_device(index)


_device_proxy: DeviceProxy | None = None


def get_device_proxy() -> DeviceProxy:
    """Get or create the global DeviceProxy singleton."""
    global _device_proxy
    if _device_proxy is None:
        _device_proxy = DeviceProxy()
    return _device_proxy


def get_proc_mesh(
    name: str,
    procs: int,
    with_gpus: bool,
    bootstrap: Callable | None = None,
):
    """Create a Monarch ProcMesh with the requested resources.

    Parameters
    ----------
    name
        Unique name for the ProcMesh.
    procs
        Number of processes to spawn.
    with_gpus
        Whether processes need accelerator devices.
    bootstrap
        Optional function run in each spawned process before the actor
        is constructed.
    """
    from monarch._src.actor.host_mesh import this_host

    host = this_host()
    proxy = get_device_proxy()

    per_host = {}
    if with_gpus:
        per_host[proxy.resource_key] = procs
    else:
        per_host["cpu"] = procs

    if bootstrap is None and not with_gpus:
        bootstrap = make_cpu_bootstrap()

    return host.spawn_procs(
        per_host=per_host,
        bootstrap=bootstrap,
        name=name,
    )


def ensure_ascend_custom_opp_path() -> None:
    """Pre-set ``ASCEND_CUSTOM_OPP_PATH`` for Monarch-spawned processes."""
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        return
    try:
        import vllm_ascend

        pkg_dir = os.path.dirname(os.path.realpath(vllm_ascend.__file__))
        custom_opp = os.path.join(pkg_dir, "_cann_ops_custom", "vendors", "vllm-ascend")
        if os.path.isdir(custom_opp):
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp
            logger.info("Set ASCEND_CUSTOM_OPP_PATH=%s", custom_opp)
    except ImportError:
        pass


def make_generator_bootstrap(device_ids_str: str) -> Callable:
    """Bootstrap for generator actors: make devices visible."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    proxy = get_device_proxy()
    device_env = proxy.device_control_env_var

    def _bootstrap():
        if device_env:
            os.environ[device_env] = device_ids_str
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path

    return _bootstrap


def make_cpu_bootstrap() -> Callable:
    """Bootstrap for CPU-only actors."""

    def _bootstrap():
        pass

    return _bootstrap


def make_trainer_bootstrap_single(device_id: int) -> Callable:
    """Bootstrap for single-process training: bind to one device."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    proxy = get_device_proxy()
    device_env = proxy.device_control_env_var

    def _bootstrap():
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
        if device_env:
            os.environ[device_env] = str(device_id)
        _init_device(0)

    return _bootstrap


def make_trainer_bootstrap_multi(all_device_ids: str) -> Callable:
    """Bootstrap for multi-process FSDP training: all devices visible."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    proxy = get_device_proxy()
    device_env = proxy.device_control_env_var

    def _bootstrap():
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
        if device_env:
            os.environ[device_env] = all_device_ids
        _init_device()

    return _bootstrap


def _init_device(device_index: int | None = None) -> None:
    """Import the torch backend and optionally set device."""
    import torch

    proxy = get_device_proxy()
    if proxy.device_type == "npu":
        import torch_npu  # noqa: F401

        if device_index is not None:
            torch.npu.set_device(device_index)
    elif proxy.device_type == "cuda":
        if device_index is not None:
            torch.cuda.set_device(device_index)
