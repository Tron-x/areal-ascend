"""GroupBuffer -- GRPO-style group rollout management with Windowed FIFO.

Collects individual episodes into groups (same prompt, multiple responses)
and only releases complete groups for training.

Key features:
- Episode grouping: accumulate until ``group_size`` responses arrive
- FIFO ordering: complete groups dequeued oldest-first
- Staleness expiry: incomplete groups too old are dropped
- GroupFilter: optionally drop untrainable groups (all same reward)
- **Windowed FIFO** (MiniMax Forge): training scheduler can only see
  groups within a sliding window of the generation head, preventing
  data distribution shift toward fast/easy samples.

Windowed FIFO (from MiniMax Forge blog):
    The training scheduler is restricted to a visibility window of
    size ``window_size``. Only complete groups whose ``create_step``
    is within ``[head, head + window_size)`` can be sampled.
    Groups outside the window are blocked even if complete.
    This prevents fast tasks from dominating the training distribution.

Usage::

    buf = await GroupBuffer.options(procs=1).as_actor(
        group_size=4, max_staleness_steps=2, window_size=0.3,
    )
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


@dataclass
class GroupEntry:
    """A group of episodes for the same prompt, accumulating until complete."""

    group_id: str
    episodes: list[dict[str, Any]] = field(default_factory=list)
    version: int = -1
    create_step: int = -1
    create_time: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return False  # set by GroupBuffer based on group_size


class GroupBuffer(ForgeActor):
    """Monarch actor that collects episodes into GRPO groups.

    Episodes arrive asynchronously (one at a time from rollout producers).
    They are grouped by ``group_id``.  Once a group has ``group_size``
    episodes, it becomes available for consumption.

    Args:
        group_size: Number of episodes per group (GRPO ``n_samples``).
        max_groups: Maximum number of pending groups before oldest is dropped.
        max_staleness_steps: Incomplete groups older than this many training
            steps are expired.
        group_filter: Optional callable ``(group_id, episodes) -> bool``.
            Return True to **drop** the group (e.g. all-correct or all-wrong).
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        group_size: int = 4,
        max_groups: int = 1024,
        max_staleness_steps: int = 2,
        group_filter: Callable[[str, list[dict]], bool] | None = None,
        window_size: float = 1.0,
    ):
        self._group_size = group_size
        self._max_groups = max_groups
        self._max_staleness_steps = max_staleness_steps
        self._filter = group_filter
        self._window_size = window_size

        self._pending: OrderedDict[str, GroupEntry] = OrderedDict()
        self._complete: OrderedDict[str, GroupEntry] = OrderedDict()

        self._generation_head: int = 0
        self._total_episodes = 0
        self._total_groups_completed = 0
        self._total_groups_expired = 0
        self._total_groups_filtered = 0
        self._total_window_blocked = 0

    @endpoint
    def add_episode(
        self,
        group_id: str,
        episode: dict[str, Any],
        version: int = -1,
        step: int = -1,
    ) -> dict:
        """Add a single episode to its group.

        Args:
            group_id: Identifier linking episodes from the same prompt.
            episode: Episode data dict.
            version: Policy version that generated this episode.
            step: Current global training step.

        Returns:
            Status dict with group progress.
        """
        self._total_episodes += 1

        if group_id not in self._pending:
            if len(self._pending) >= self._max_groups:
                oldest_id = next(iter(self._pending))
                self._pending.pop(oldest_id)
                self._total_groups_expired += 1

            self._pending[group_id] = GroupEntry(
                group_id=group_id,
                version=version,
                create_step=step,
                create_time=time.time(),
            )

        entry = self._pending[group_id]
        entry.episodes.append(episode)

        if len(entry.episodes) >= self._group_size:
            self._pending.pop(group_id)

            if self._filter and self._filter(group_id, entry.episodes):
                self._total_groups_filtered += 1
            else:
                self._complete[group_id] = entry
                self._total_groups_completed += 1

        return {
            "group_id": group_id,
            "group_progress": len(entry.episodes)
            if group_id in self._pending
            else self._group_size,
            "group_size": self._group_size,
            "pending_groups": len(self._pending),
            "complete_groups": len(self._complete),
        }

    @endpoint
    def add_group(
        self,
        group_id: str,
        episodes: list[dict[str, Any]],
        version: int = -1,
        step: int = -1,
    ) -> dict:
        """Add a pre-assembled complete group (for sync pipelines).

        Bypasses the accumulation logic -- useful when the rollout
        producer already generates all ``n_samples`` responses together.
        """
        if self._filter and self._filter(group_id, episodes):
            self._total_groups_filtered += 1
            return {"filtered": True, "group_id": group_id}

        entry = GroupEntry(
            group_id=group_id,
            episodes=episodes,
            version=version,
            create_step=step,
        )
        self._complete[group_id] = entry
        self._total_episodes += len(episodes)
        self._total_groups_completed += 1

        return {
            "group_id": group_id,
            "complete_groups": len(self._complete),
        }

    @endpoint
    def sample_group(
        self,
        current_step: int = -1,
    ) -> list[dict[str, Any]] | None:
        """Dequeue the oldest complete group within the visibility window.

        Windowed FIFO: only groups whose ``create_step`` falls within
        ``[head, head + window)`` are visible to the trainer.  This
        prevents fast/easy samples from dominating the training batch.

        Returns:
            List of episode dicts (length = group_size), or None.
        """
        if current_step >= 0:
            self._expire_stale(current_step)

        if not self._complete:
            return None

        window = self._compute_window()

        for group_id in list(self._complete):
            entry = self._complete[group_id]
            if entry.create_step <= window:
                self._complete.pop(group_id)
                self._advance_head(entry.create_step)
                return entry.episodes
            self._total_window_blocked += 1

        return None

    @endpoint
    def sample_groups(
        self,
        count: int = 1,
        current_step: int = -1,
    ) -> list[list[dict[str, Any]]] | None:
        """Dequeue multiple complete groups within the visibility window.

        Returns:
            List of groups, or None if none available in the window.
        """
        if current_step >= 0:
            self._expire_stale(current_step)

        if not self._complete:
            return None

        result = []
        for _ in range(min(count, len(self._complete))):
            window = self._compute_window()
            found = False
            for group_id in list(self._complete):
                entry = self._complete[group_id]
                if entry.create_step <= window:
                    self._complete.pop(group_id)
                    self._advance_head(entry.create_step)
                    result.append(entry.episodes)
                    found = True
                    break
            if not found:
                break

        return result if result else None

    @endpoint
    def get_stats(self) -> dict:
        return {
            "pending_groups": len(self._pending),
            "complete_groups": len(self._complete),
            "group_size": self._group_size,
            "window_size": self._window_size,
            "generation_head": self._generation_head,
            "total_episodes": self._total_episodes,
            "total_groups_completed": self._total_groups_completed,
            "total_groups_expired": self._total_groups_expired,
            "total_groups_filtered": self._total_groups_filtered,
            "total_window_blocked": self._total_window_blocked,
        }

    @endpoint
    def clear(self) -> None:
        self._pending.clear()
        self._complete.clear()

    def _compute_window(self) -> int:
        """Compute the upper bound of the visibility window.

        Window = head + int(total_complete * window_size).
        When ``window_size=1.0``, all complete groups are visible (no blocking).
        When ``window_size=0.3``, only the first 30% of groups are visible.
        """
        total = len(self._complete)
        if total == 0:
            return self._generation_head
        w = max(1, int(total * self._window_size))
        steps = sorted(e.create_step for e in self._complete.values())
        return steps[min(w - 1, len(steps) - 1)]

    def _advance_head(self, consumed_step: int) -> None:
        """Advance the generation head after consuming a group."""
        self._generation_head = max(self._generation_head, consumed_step + 1)

    def _expire_stale(self, current_step: int) -> None:
        """Remove incomplete groups that are too old."""
        expired = []
        for gid, entry in self._pending.items():
            if current_step - entry.create_step > self._max_staleness_steps:
                expired.append(gid)

        for gid in expired:
            self._pending.pop(gid)
            self._total_groups_expired += 1

        if expired:
            logger.debug(f"[GroupBuffer] Expired {len(expired)} stale groups")
