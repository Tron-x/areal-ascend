"""Load-balancing routers for service request routing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from forge.service.replica import Replica

logger = logging.getLogger(__name__)


class Router(ABC):
    """Abstract base class for routing logic."""

    @abstractmethod
    def get_replica(
        self,
        healthy_replicas: list[Replica],
        sess_id: str | None = None,
        session_map: dict[str, int] | None = None,
    ) -> Replica:
        pass


class RoundRobinRouter(Router):
    """Stateless round-robin request distribution."""

    def __init__(self):
        self._next_idx = 0

    def get_replica(
        self,
        healthy_replicas: list[Replica],
        sess_id: str | None = None,
        session_map: dict[str, int] | None = None,
    ) -> Replica:
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available for load balancing")
        self._next_idx = (self._next_idx + 1) % len(healthy_replicas)
        return healthy_replicas[self._next_idx]


class LeastLoadedRouter(Router):
    """Route to the replica with the lowest current load."""

    def get_replica(
        self,
        healthy_replicas: list[Replica],
        sess_id: str | None = None,
        session_map: dict[str, int] | None = None,
    ) -> Replica:
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available")
        return min(healthy_replicas, key=lambda r: r.current_load)


class SessionRouter(Router):
    """Sticky session routing with fallback for new/orphaned sessions.

    Critical for agentic RL where multi-turn conversations must stay
    on the same Generator replica to maintain KV cache locality.
    """

    def __init__(self, fallback_router: Router):
        self.fallback_router = fallback_router

    def get_replica(
        self,
        healthy_replicas: list[Replica],
        sess_id: str | None = None,
        session_map: dict[str, int] | None = None,
    ) -> Replica:
        if sess_id is None:
            raise ValueError("SessionRouter requires a session ID")
        if session_map is None:
            raise ValueError("Session map must be provided for SessionRouter")

        if sess_id in session_map:
            replica_idx = session_map[sess_id]
            for r in healthy_replicas:
                if r.idx == replica_idx:
                    return r
            del session_map[sess_id]

        replica = self.fallback_router.get_replica(
            healthy_replicas, sess_id, session_map
        )
        session_map[sess_id] = replica.idx
        logger.debug("Assigning session %s to replica %d", sess_id, replica.idx)
        return replica
