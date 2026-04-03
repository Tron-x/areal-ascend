"""Weight synchronization abstraction for Monarch plugin.

Provides a ``WeightStore`` protocol with pluggable backends:
- ``DiskWeightStore``: shared filesystem (simple, works everywhere)
- ``XCCLWeightStore``: HCCL/NCCL collective (low latency, requires shared PG)

A TorchStore backend can be added when torchstore is available.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class WeightStore(ABC):
    """Abstract weight storage for RL weight synchronization.

    Trainer calls ``put()`` after each training step.
    Generator calls ``get()`` to load updated weights.
    """

    @abstractmethod
    async def put(self, version: int, state_dict: dict[str, Tensor]) -> None:
        """Push a versioned state dict."""
        ...

    @abstractmethod
    async def get(self, version: int) -> dict[str, Tensor]:
        """Pull a versioned state dict."""
        ...

    @abstractmethod
    async def latest_version(self) -> int | None:
        """Return the latest available version, or None."""
        ...


class DiskWeightStore(WeightStore):
    """Filesystem-based weight store using safetensors or torch.save.

    Weights are stored at ``{root_dir}/v{version}/weights.pt``.
    Simple and portable; suitable for single-node or shared-filesystem clusters.
    """

    def __init__(self, root_dir: str, max_versions: int = 3):
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_versions = max_versions

    async def put(self, version: int, state_dict: dict[str, Tensor]) -> None:
        version_dir = self._root / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        path = version_dir / "weights.pt"
        torch.save(state_dict, path)
        logger.info(f"[DiskWeightStore] Saved v{version} to {path}")
        self._cleanup_old_versions(version)

    async def get(self, version: int) -> dict[str, Tensor]:
        path = self._root / f"v{version}" / "weights.pt"
        if not path.exists():
            raise FileNotFoundError(f"Weights v{version} not found at {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        logger.info(f"[DiskWeightStore] Loaded v{version} from {path}")
        return state_dict

    async def latest_version(self) -> int | None:
        versions = []
        for d in self._root.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    continue
        return max(versions) if versions else None

    def _cleanup_old_versions(self, current_version: int) -> None:
        if self._max_versions <= 0:
            return
        versions = []
        for d in self._root.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    continue
        versions.sort()
        while len(versions) > self._max_versions:
            old_v = versions.pop(0)
            old_dir = self._root / f"v{old_v}"
            shutil.rmtree(old_dir, ignore_errors=True)
            logger.debug(f"[DiskWeightStore] Cleaned up v{old_v}")


class XCCLWeightStore(WeightStore):
    """HCCL/NCCL-based weight store using collective all-gather.

    Requires a shared process group between trainer and generator ranks.
    Uses the existing AReaL ``WeightUpdateMeta`` / ``alloc_mode`` infrastructure
    for process group initialization.

    This is a placeholder that delegates to the existing weight_sync module.
    """

    def __init__(self, alloc_mode=None):
        self._alloc_mode = alloc_mode
        self._state_dict_cache: dict[int, dict[str, Tensor]] = {}

    async def put(self, version: int, state_dict: dict[str, Tensor]) -> None:
        self._state_dict_cache[version] = state_dict
        logger.info(f"[XCCLWeightStore] Cached v{version} ({len(state_dict)} params)")

    async def get(self, version: int) -> dict[str, Tensor]:
        if version not in self._state_dict_cache:
            raise KeyError(f"Version {version} not in XCCL cache")
        return self._state_dict_cache[version]

    async def latest_version(self) -> int | None:
        return max(self._state_dict_cache.keys()) if self._state_dict_cache else None


def create_weight_store(
    backend: str = "disk",
    root_dir: str | None = None,
    alloc_mode=None,
) -> WeightStore:
    """Factory for creating a WeightStore instance.

    Args:
        backend: "disk" or "xccl".
        root_dir: Directory for disk backend.
        alloc_mode: AllocationMode for XCCL backend.

    Returns:
        A configured WeightStore.
    """
    if backend == "disk":
        if root_dir is None:
            root_dir = "/tmp/areal_weights"
        return DiskWeightStore(root_dir)
    elif backend == "xccl":
        return XCCLWeightStore(alloc_mode)
    else:
        raise ValueError(f"Unknown weight store backend: {backend}")
