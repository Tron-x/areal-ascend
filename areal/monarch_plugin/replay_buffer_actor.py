"""ReplayBufferActor -- Monarch actor for experience replay.

Phase 6: Decouples rollout production from training consumption,
enabling pipeline parallelism where rollout step N+1 overlaps with
training on batch N.

Architecture:
  ReplayBufferActor runs on its own CPU ProcMesh.  The rollout loop
  pushes completed batches via ``add_batch()``, and the training loop
  pulls them via ``sample_batch()``.  Version-aware staleness control
  ensures off-policy data is bounded.

  Rollout coroutine ──add_batch──→ ReplayBufferActor ──sample_batch──→ Training coroutine
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from monarch.actor import endpoint

from areal.monarch_plugin.actor import AReaLMonarchActor

logger = logging.getLogger(__name__)


class ReplayBufferActor(AReaLMonarchActor):
    """Monarch Actor that buffers rollout batches for async training.

    Each entry is a dict with:
      - ``"batch"``: serialised rollout batch (dict of lists/numpy)
      - ``"policy_version"``: the model version used during rollout
      - ``"timestamp"``: wall-clock time when the batch was added

    Staleness control:
      ``sample_batch(current_version, max_staleness)`` discards entries
      whose ``policy_version < current_version - max_staleness``.
    """

    procs = 1
    with_gpus = False

    def __init__(self, max_size: int = 16):
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._max_size = max_size
        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0

    @endpoint
    async def add_batch(self, batch: dict, policy_version: int) -> dict:
        """Push a rollout batch into the buffer.

        Parameters
        ----------
        batch : dict
            Serialised rollout batch (dict of lists or numpy arrays).
        policy_version : int
            Model version that produced this batch.

        Returns
        -------
        dict
            Buffer status after insertion.
        """
        entry = {
            "batch": batch,
            "policy_version": policy_version,
            "timestamp": time.monotonic(),
        }
        self._buffer.append(entry)
        self._total_added += 1

        if len(self._buffer) == self._max_size:
            logger.debug(
                f"[ReplayBuffer] Buffer full ({self._max_size}), "
                f"oldest entry will be evicted on next add"
            )

        return {
            "size": len(self._buffer),
            "policy_version": policy_version,
        }

    @endpoint
    async def sample_batch(
        self,
        current_version: int = -1,
        max_staleness: int = -1,
    ) -> dict | None:
        """Pop the oldest non-stale batch from the buffer.

        Parameters
        ----------
        current_version : int
            Current training policy version (for staleness check).
            -1 disables staleness filtering.
        max_staleness : int
            Maximum allowed ``current_version - policy_version``.
            -1 disables staleness filtering.

        Returns
        -------
        dict or None
            The batch dict, or None if buffer is empty / all entries stale.
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
                    f"[ReplayBuffer] Evicted stale batch "
                    f"(version={entry['policy_version']}, "
                    f"current={current_version}, "
                    f"staleness={current_version - entry['policy_version']})"
                )
                continue

            self._buffer.popleft()
            self._total_sampled += 1
            return entry["batch"]

        return None

    @endpoint
    async def buffer_size(self) -> int:
        """Return the number of batches currently in the buffer."""
        return len(self._buffer)

    @endpoint
    async def get_stats(self) -> dict:
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

    @endpoint
    async def shutdown(self) -> None:
        logger.info(
            f"[ReplayBuffer] Shutting down. "
            f"added={self._total_added}, sampled={self._total_sampled}, "
            f"evicted={self._total_evicted}"
        )
        self._buffer.clear()
