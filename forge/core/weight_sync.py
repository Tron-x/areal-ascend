"""Weight synchronization abstractions for Forge.

Provides ``WeightStore`` ABC and a filesystem-based implementation.
Backend-specific implementations (XCCL, NCCL, TorchStore) live in adapters.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger("forge.weight_sync")


class WeightStore(ABC):
    """Abstract interface for weight synchronization between trainer and generator.

    The trainer calls ``push(version)`` after updating weights.
    The generator calls ``pull(version)`` to load the new weights.
    """

    @abstractmethod
    async def push(self, version: int) -> None:
        """Push weights with the given version identifier."""

    @abstractmethod
    async def pull(self, version: int) -> None:
        """Pull (load) weights for the given version."""

    @abstractmethod
    async def latest_version(self) -> int:
        """Return the latest available weight version."""


class DiskWeightStore(WeightStore):
    """Filesystem-based weight store.

    Weights are saved as PyTorch ``state_dict`` files under
    ``{base_path}/v{version}/``. A ``latest.json`` file tracks the
    current version.

    Suitable for shared filesystems (NFS, Lustre) in cluster environments.
    """

    def __init__(self, base_path: str, save_fn=None, load_fn=None) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._save_fn = save_fn
        self._load_fn = load_fn

    def _version_dir(self, version: int) -> Path:
        return self._base / f"v{version}"

    def _meta_path(self) -> Path:
        return self._base / "latest.json"

    async def push(self, version: int) -> None:
        vdir = self._version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)

        if self._save_fn is not None:
            self._save_fn(str(vdir))

        meta = {"version": version, "timestamp": time.time()}
        self._meta_path().write_text(json.dumps(meta))
        logger.info("DiskWeightStore: pushed version %d to %s", version, vdir)

    async def pull(self, version: int) -> None:
        vdir = self._version_dir(version)
        if not vdir.exists():
            raise FileNotFoundError(f"Weight version {version} not found at {vdir}")

        if self._load_fn is not None:
            self._load_fn(str(vdir))

        logger.info("DiskWeightStore: pulled version %d from %s", version, vdir)

    async def latest_version(self) -> int:
        meta_path = self._meta_path()
        if not meta_path.exists():
            return -1
        meta = json.loads(meta_path.read_text())
        return meta.get("version", -1)

    def cleanup(self, keep_versions: int = 2) -> None:
        """Remove old weight versions, keeping the N most recent."""
        versions = sorted(
            (
                int(d.name[1:])
                for d in self._base.iterdir()
                if d.is_dir() and d.name.startswith("v")
            ),
            reverse=True,
        )
        for v in versions[keep_versions:]:
            vdir = self._version_dir(v)
            shutil.rmtree(vdir, ignore_errors=True)
            logger.debug("DiskWeightStore: cleaned up version %d", v)
