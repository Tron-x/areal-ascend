"""Platform-agnostic bootstrap factories for Monarch actor processes.

Each factory returns a callable that runs in the spawned process *before*
the actor class is instantiated.  This is where device visibility, library
paths, and multiprocessing settings are configured.

Currently supports:
  - Ascend NPU (torch_npu / HCCL)
  - CUDA (torch / NCCL)  -- placeholder, ready for implementation
"""

from __future__ import annotations

import os
from collections.abc import Callable

from areal.infra.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger("MonarchBootstraps")


# ---------------------------------------------------------------------------
# CANN / Ascend helpers
# ---------------------------------------------------------------------------


def ensure_ascend_custom_opp_path() -> None:
    """Pre-set ``ASCEND_CUSTOM_OPP_PATH`` for Monarch-spawned processes.

    vllm_ascend ships custom CANN ops under ``_cann_ops_custom``.  The CANN
    runtime reads this env-var at startup.  Normally vllm_ascend sets it
    during its platform init, but Monarch worker processes are forked from
    the host agent which never imports vllm_ascend.  Setting it here ensures
    all child processes inherit the value.
    """
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        return
    try:
        import vllm_ascend

        pkg_dir = os.path.dirname(os.path.realpath(vllm_ascend.__file__))
        custom_opp = os.path.join(pkg_dir, "_cann_ops_custom", "vendors", "vllm-ascend")
        if os.path.isdir(custom_opp):
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp
            logger.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Bootstrap factories
# ---------------------------------------------------------------------------


def make_generator_bootstrap(device_ids_str: str) -> Callable:
    """Bootstrap for GeneratorActor: make devices visible but do NOT
    initialise a device context.  AsyncLLM only validates config; actual
    compute happens on workers spawned by AReaLMonarchExecutor.

    Also propagates LD_LIBRARY_PATH for vendor runtime libraries.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    device_env = current_platform.device_control_env_var

    def _bootstrap():
        os.environ[device_env] = device_ids_str
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path

    return _bootstrap


def make_cpu_bootstrap() -> Callable:
    """Bootstrap for CPU-only actors (Reward, Sandbox, Agent, etc.)."""

    def _bootstrap():
        pass

    return _bootstrap


def make_trainer_bootstrap_single(device_id: int) -> Callable:
    """Bootstrap for single-process training: bind to one device.

    Forces ``fork`` multiprocessing for AReaL DataLoader compatibility.
    Propagates LD_LIBRARY_PATH for runtime library availability.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    device_env = current_platform.device_control_env_var

    def _bootstrap():
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
        os.environ[device_env] = str(device_id)
        _init_device(0)

    return _bootstrap


def make_trainer_bootstrap_multi(all_device_ids: str) -> Callable:
    """Bootstrap for multi-process FSDP training: all training devices visible.

    Each spawned process sees ALL training devices.  The actual device
    selection (``torch.{npu,cuda}.set_device(local_rank)``) happens inside
    ``TrainerActor.initialize()`` after the Monarch rank is known.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    device_env = current_platform.device_control_env_var

    def _bootstrap():
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
        os.environ[device_env] = all_device_ids
        _init_device()

    return _bootstrap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_device(device_index: int | None = None) -> None:
    """Import the appropriate torch backend and optionally set device.

    On Ascend: imports torch_npu so the NPU backend is registered.
    On CUDA: plain torch is sufficient.
    """
    import torch

    platform = current_platform
    if platform.device_type == "npu":
        import torch_npu  # noqa: F401

        if device_index is not None:
            torch.npu.set_device(device_index)
    elif platform.device_type == "cuda":
        if device_index is not None:
            torch.cuda.set_device(device_index)
