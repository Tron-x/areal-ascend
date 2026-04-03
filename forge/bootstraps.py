"""Platform-agnostic bootstrap factories for Monarch actor processes.

Each factory returns a callable that runs in the spawned process *before*
the actor class is instantiated. This is where device visibility, library
paths, and multiprocessing settings are configured.

Currently supports:
  - Ascend NPU (torch_npu / HCCL)
  - CUDA (torch / NCCL)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger("MonarchBootstraps")


def _detect_device_env_var() -> str:
    """Detect the correct device visibility env var for the current platform."""
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "ASCEND_RT_VISIBLE_DEVICES"
        if torch.cuda.is_available():
            return "CUDA_VISIBLE_DEVICES"
    except ImportError:
        pass
    return "CUDA_VISIBLE_DEVICES"


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
            logger.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp}")
    except ImportError:
        pass


def make_generator_bootstrap(device_ids_str: str) -> Callable:
    """Bootstrap for GeneratorActor: make devices visible."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    device_env = _detect_device_env_var()

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
    """Bootstrap for single-process training: bind to one device."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    device_env = _detect_device_env_var()

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
    """Bootstrap for multi-process FSDP training: all training devices visible."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    device_env = _detect_device_env_var()

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


def _init_device(device_index: int | None = None) -> None:
    """Import the appropriate torch backend and optionally set device."""
    import torch

    if hasattr(torch, "npu") and torch.npu.is_available():
        import torch_npu  # noqa: F401

        if device_index is not None:
            torch.npu.set_device(device_index)
    elif torch.cuda.is_available():
        if device_index is not None:
            torch.cuda.set_device(device_index)
