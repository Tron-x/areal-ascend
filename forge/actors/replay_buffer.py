"""ReplayBuffer actor with eviction, sampling, and version-aware filtering.

Serves as the bridge between the rollout producer and training consumer
in the async pipeline.  Supports:
- Age-based eviction (samples too old relative to training step)
- Count-based eviction (max buffer size)
- Version-based filtering (reject data from stale policy versions)
- Blocking wait-and-sample (avoids busy polling in consumer loop)
- Batch add (multiple items at once from rollout producer)
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    AGE = "age"
    COUNT = "count"
    NONE = "none"


@dataclass
class BufferEntry:
    data: dict[str, Any]
    version: int
    insert_time: float
    insert_step: int
    sample_count: int = 0


class ReplayBuffer(ForgeActor):
    """Asynchronous replay buffer for decoupled rollout and training.

    Supports configurable eviction policies to prevent stale experience
    from being trained on, and provides sampling that respects version
    constraints for on-policy RL.

    Deploy as a single actor::

        buffer = await ReplayBuffer.options(procs=1).as_actor(
            max_size=1024,
            eviction_policy="age",
            max_age_steps=2,
        )
        await buffer.add.call_one(batch_data, version=1)
        batch = await buffer.sample.call_one(batch_size=64, current_step=5)
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        max_size: int = 4096,
        eviction_policy: str = "count",
        max_age_steps: int = 2,
        max_sample_count: int = 4,
    ):
        self._buffer: deque[BufferEntry] = deque()
        self._max_size = max_size
        self._eviction_policy = EvictionPolicy(eviction_policy)
        self._max_age_steps = max_age_steps
        self._max_sample_count = max_sample_count
        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0

    @endpoint
    def add(
        self,
        data: dict[str, Any],
        version: int = -1,
        step: int = -1,
    ) -> dict:
        """Add experience to the buffer.

        Args:
            data: Rollout data (tensors as lists or dicts).
            version: Policy version that generated this data.
            step: Global training step at insertion time.

        Returns:
            Buffer statistics after insertion.
        """
        entry = BufferEntry(
            data=data,
            version=version,
            insert_time=time.time(),
            insert_step=step,
        )
        self._buffer.append(entry)
        self._total_added += 1

        self._evict()

        return {
            "buffer_size": len(self._buffer),
            "total_added": self._total_added,
            "total_evicted": self._total_evicted,
        }

    @endpoint
    def add_batch(
        self,
        items: list[dict[str, Any]],
        version: int = -1,
        step: int = -1,
    ) -> dict:
        """Add multiple experience entries to the buffer at once.

        Args:
            items: List of rollout data dicts.
            version: Policy version that generated this data.
            step: Global training step at insertion time.

        Returns:
            Buffer statistics after insertion.
        """
        now = time.time()
        for data in items:
            entry = BufferEntry(
                data=data,
                version=version,
                insert_time=now,
                insert_step=step,
            )
            self._buffer.append(entry)
            self._total_added += 1

        self._evict()

        return {
            "buffer_size": len(self._buffer),
            "total_added": self._total_added,
            "total_evicted": self._total_evicted,
            "batch_added": len(items),
        }

    @endpoint
    def sample(
        self,
        batch_size: int = 1,
        current_step: int = -1,
        min_version: int = -1,
    ) -> list[dict] | None:
        """Sample a batch from the buffer.

        Args:
            batch_size: Number of entries to sample.
            current_step: Current training step (for staleness filtering).
            min_version: Minimum policy version to accept (reject older data).

        Returns:
            List of data dicts, or None if buffer is empty / no valid entries.
        """
        return self._do_sample(batch_size, current_step, min_version)

    @endpoint
    def wait_and_sample(
        self,
        batch_size: int = 1,
        current_step: int = -1,
        min_version: int = -1,
    ) -> list[dict] | None:
        """Sample from the buffer, returning None only if truly empty.

        The caller should retry with a sleep if None is returned::

            while batch is None:
                batch = await buffer.wait_and_sample.call_one(...)
                await asyncio.sleep(0.5)
        """
        return self._do_sample(batch_size, current_step, min_version)

    def _do_sample(
        self,
        batch_size: int = 1,
        current_step: int = -1,
        min_version: int = -1,
    ) -> list[dict] | None:
        """Internal sampling logic shared by sample and wait_and_sample."""
        if not self._buffer:
            return None

        if current_step >= 0 and self._eviction_policy == EvictionPolicy.AGE:
            self._evict_by_age(current_step)

        if not self._buffer:
            return None

        if min_version >= 0:
            candidates = [e for e in self._buffer if e.version >= min_version]
        else:
            candidates = list(self._buffer)

        if not candidates:
            return None

        k = min(batch_size, len(candidates))
        selected = random.sample(candidates, k)

        for entry in selected:
            entry.sample_count += 1

        if self._max_sample_count > 0:
            self._evict_by_sample_count()

        self._total_sampled += k
        return [entry.data for entry in selected]

    @endpoint
    def buffer_size(self) -> int:
        return len(self._buffer)

    @endpoint
    def clear(self) -> None:
        self._buffer.clear()

    @endpoint
    def get_stats(self) -> dict:
        versions = [e.version for e in self._buffer]
        return {
            "buffer_size": len(self._buffer),
            "total_added": self._total_added,
            "total_sampled": self._total_sampled,
            "total_evicted": self._total_evicted,
            "min_version": min(versions) if versions else -1,
            "max_version": max(versions) if versions else -1,
        }

    def _evict(self):
        """Apply configured eviction policy."""
        if self._eviction_policy == EvictionPolicy.COUNT:
            self._evict_by_count()

    def _evict_by_count(self):
        while len(self._buffer) > self._max_size:
            self._buffer.popleft()
            self._total_evicted += 1

    def _evict_by_age(self, current_step: int):
        evicted = 0
        while self._buffer:
            oldest = self._buffer[0]
            if current_step - oldest.insert_step > self._max_age_steps:
                self._buffer.popleft()
                evicted += 1
            else:
                break
        self._total_evicted += evicted
        if evicted > 0:
            logger.debug(f"[ReplayBuffer] Evicted {evicted} stale entries")

    def _evict_by_sample_count(self):
        evicted = 0
        new_buffer = deque()
        for entry in self._buffer:
            if entry.sample_count < self._max_sample_count:
                new_buffer.append(entry)
            else:
                evicted += 1
        self._buffer = new_buffer
        self._total_evicted += evicted
