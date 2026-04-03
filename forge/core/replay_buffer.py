"""Framework-agnostic ReplayBuffer for Forge.

Pure Python implementation -- no Monarch or Ray dependency.
Supports version-aware staleness control and FIFO eviction.

Extracted from ``areal/monarch_plugin/replay_buffer_actor.py`` and
stripped of all Monarch actor decorators.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from forge.api.types import TrainBatch

logger = logging.getLogger("forge.replay_buffer")


class ReplayBuffer:
    """In-memory experience replay buffer with staleness control.

    Architecture::

        Rollout loop ──add()──→ ReplayBuffer ──sample()──→ Training loop

    Each entry stores:
      - ``batch``: the rollout data
      - ``policy_version``: model version that produced this batch
      - ``timestamp``: wall-clock time when the batch was added
    """

    def __init__(self, max_size: int = 16) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._max_size = max_size
        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0

    def add(self, batch: TrainBatch | dict, policy_version: int = -1) -> dict:
        """Push a rollout batch into the buffer.

        Returns buffer status after insertion.
        """
        data = batch.tensors if isinstance(batch, TrainBatch) else batch
        entry = {
            "batch": data,
            "policy_version": policy_version,
            "timestamp": time.monotonic(),
        }
        self._buffer.append(entry)
        self._total_added += 1
        return {"size": len(self._buffer), "policy_version": policy_version}

    def sample(
        self,
        current_version: int = -1,
        max_staleness: int = -1,
    ) -> dict | None:
        """Pop the oldest non-stale batch from the buffer.

        Parameters
        ----------
        current_version
            Current training policy version for staleness check.
            -1 disables staleness filtering.
        max_staleness
            Maximum allowed ``current_version - policy_version``.
            -1 disables staleness filtering.
        """
        while self._buffer:
            entry = self._buffer[0]
            if (
                max_staleness >= 0
                and current_version >= 0
                and current_version - entry["policy_version"] > max_staleness
            ):
                self._buffer.popleft()
                self._total_evicted += 1
                logger.debug(
                    "Evicted stale batch (version=%d, current=%d)",
                    entry["policy_version"],
                    current_version,
                )
                continue

            self._buffer.popleft()
            self._total_sampled += 1
            return entry["batch"]
        return None

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def stats(self) -> dict:
        """Return buffer statistics."""
        versions = [e["policy_version"] for e in self._buffer]
        return {
            "current_size": len(self._buffer),
            "max_size": self._max_size,
            "total_added": self._total_added,
            "total_sampled": self._total_sampled,
            "total_evicted": self._total_evicted,
            "version_range": (
                f"{min(versions)}-{max(versions)}" if versions else "empty"
            ),
        }

    def clear(self) -> None:
        """Remove all entries."""
        self._buffer.clear()
