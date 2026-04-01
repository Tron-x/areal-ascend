"""Declarative actor specification types for Monarch orchestration.

Each actor class inherits from :class:`MonarchActor` (in ``actor_base.py``)
and declares its resource requirements, dependencies, and lifecycle hooks as
class-level attributes.  The :class:`ActorRegistry` discovers these via
``cls.actor_name()`` and ``cls.dependencies``, resolves the dependency graph,
and calls ``cls.spawn()`` / ``cls.initialize_actor()`` / ``cls.shutdown_actor()``
in the correct order.

This module provides the shared data types:
  - :class:`ResourceKind` — CPU, NPU_SINGLE, NPU_MULTI
  - :class:`ActorRef` / :class:`CtxRef` — declarative reference markers
  - :class:`ActorContext` — shared mutable state passed to all classmethods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceKind(Enum):
    """What kind of ProcMesh to create for an actor."""

    CPU = "cpu"  # per_host={"cpu": 1}
    NPU_SINGLE = "npu_single"  # per_host={"npu": 1}, single device visible
    NPU_MULTI = "npu_multi"  # per_host={"npu": N}, all training devices visible


# ---------------------------------------------------------------------------
# Declarative reference types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorRef:
    """Reference to another actor by name.

    Resolved at spawn / init time to the actual actor reference from
    the already-spawned actors dict.
    """

    name: str


@dataclass(frozen=True)
class CtxRef:
    """Reference to a value in :attr:`ActorContext.extra`.

    Resolved at spawn / init time.
    """

    key: str


@dataclass
class ActorContext:
    """Mutable shared state passed to all actor classmethods.

    The registry and ``MonarchActor.spawn()`` / ``initialize_actor()`` /
    ``shutdown_actor()`` read and write into this object.
    """

    config: Any
    alloc_mode: Any
    placement: Any
    host: Any
    actors: dict[str, Any] = field(default_factory=dict)
    procs: dict[str, Any] = field(default_factory=dict)
    extra_actors: dict[str, Any] = field(default_factory=dict)
    master_port: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
