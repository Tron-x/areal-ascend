"""GeneratorService -- transparent proxy for multi-replica generators.

When ``num_generator_replicas > 1``, the ``ActorRegistry`` spawns N
``GeneratorActor`` instances (one per replica) and wraps them in this
proxy.  Consumers (``MonarchVLLMEngine``, ``TrainerActor``) continue
to use ``ctx.actors["generator"]`` without code changes.

Routing strategy:

  - Inference endpoints (``/v1/completions``, ``/v1/chat/completions``):
    round-robin across replicas.
  - Weight-sync and control endpoints: fanout to **all** replicas
    (sequential).
  - Shutdown: fanout.

For ``init_weights_update_group``, the ``rank_offset`` in the payload
is adjusted per replica so that each replica's vLLM workers receive the
correct offset within the shared HCCL group.

Thread safety
-------------
``_route`` is protected by a lock so concurrent callers (multiple
TrainerActor ranks each running their own ``ThreadPoolExecutor``) get
correct round-robin behaviour.

``_fanout`` is **synchronous** -- it blocks until every replica
responds.  It is only safe for synchronous handler methods on the
actor (weight-sync, pause/resume).  Do **not** fanout to async
endpoints.
"""

from __future__ import annotations

import threading
from typing import Any

from areal.monarch_plugin.topology import ReplicaPlacement
from areal.utils import logging as areal_logging

logger = areal_logging.getLogger("GeneratorService")

# ---------------------------------------------------------------------------
# Endpoint classification
# ---------------------------------------------------------------------------

_WEIGHT_SYNC_ENDPOINTS = frozenset(
    {
        "/areal_init_weights_update_group",
        "/areal_set_update_weight_meta",
        "/areal_set_update_weight_meta_lora",
        "/areal_update_weights_xccl",
        "/areal_update_weights_lora_xccl",
        "/areal_update_weights",
        "/areal_pause_generation",
        "/areal_continue_generation",
        "/health",
    }
)

_INFERENCE_ENDPOINTS = frozenset(
    {
        "/v1/completions",
        "/v1/chat/completions",
    }
)


# ---------------------------------------------------------------------------
# Future helpers
# ---------------------------------------------------------------------------


class _SyncFuture:
    """Future-like wrapper around an already-computed result.

    Supports both ``.get()`` (blocking) and ``await`` (async) usage so
    callers that expect a Monarch ``Future`` can use it unchanged.
    """

    def __init__(self, result: Any):
        self._result = result

    def get(self, timeout: float | None = None) -> Any:
        return self._result

    def __await__(self):
        return self._async_result().__await__()

    async def _async_result(self):
        return self._result


# ---------------------------------------------------------------------------
# Attribute proxies  (duck-type Monarch actor-ref attributes)
# ---------------------------------------------------------------------------


class _HandleRequestProxy:
    """Proxies ``generator.handle_request.call_one(ep, payload)``."""

    def __init__(self, service: GeneratorService):
        self._service = service

    def call_one(self, ep: str, payload: dict) -> Any:
        svc = self._service

        if ep in _INFERENCE_ENDPOINTS:
            return svc._route(ep, payload)

        if ep in _WEIGHT_SYNC_ENDPOINTS:
            return svc._fanout(ep, payload)

        # Default: round-robin
        return svc._route(ep, payload)


class _ShutdownProxy:
    """Proxies ``generator.shutdown.call_one()`` / ``.call()``."""

    def __init__(self, service: GeneratorService):
        self._service = service

    async def call_one(self) -> None:
        """Shutdown all replicas sequentially."""
        for name, ref in self._service._replicas.items():
            try:
                await ref.shutdown.call_one()
            except Exception as e:
                logger.warning(f"Error shutting down replica '{name}': {e}")

    async def call(self) -> None:
        """Broadcast shutdown to all replicas (alias for ``call_one``)."""
        await self.call_one()


# ---------------------------------------------------------------------------
# GeneratorService
# ---------------------------------------------------------------------------


class GeneratorService:
    """Transparent proxy for multi-replica ``GeneratorActor`` instances.

    Stored in ``ctx.actors["generator"]`` so that every consumer
    (``MonarchVLLMEngine``, ``TrainerActor``, shutdown logic) sees a
    single generator reference without code changes.
    """

    def __init__(
        self,
        replicas: dict[str, Any],
        placements: list[ReplicaPlacement],
        per_replica_workers: int = 1,
    ):
        self._replicas: dict[str, Any] = replicas
        self._replica_list: list[Any] = list(replicas.values())
        self._placements = placements
        self._per_replica_workers = per_replica_workers
        self._rr_index: int = 0
        self._rr_lock = threading.Lock()

    @property
    def handle_request(self) -> _HandleRequestProxy:
        return _HandleRequestProxy(self)

    @property
    def shutdown(self) -> _ShutdownProxy:
        return _ShutdownProxy(self)

    @property
    def replicas(self) -> dict[str, Any]:
        return dict(self._replicas)

    @property
    def num_replicas(self) -> int:
        return len(self._replicas)

    # ------------------------------------------------------------------
    # Routing: round-robin to one replica
    # ------------------------------------------------------------------

    def _route(self, ep: str, payload: dict) -> Any:
        """Round-robin to one replica.  Returns a Monarch Future."""
        with self._rr_lock:
            idx = self._rr_index % len(self._replica_list)
            self._rr_index += 1
        ref = self._replica_list[idx]
        return ref.handle_request.call_one(ep, payload)

    # ------------------------------------------------------------------
    # Fanout: sequential broadcast to all replicas
    # ------------------------------------------------------------------

    def _fanout(self, ep: str, payload: dict) -> _SyncFuture:
        """Fanout to all replicas sequentially, adjusting rank_offset.

        **Sync-only**: this method blocks until every replica responds.
        Only safe for synchronous handler methods on the actor
        (weight-sync, pause/resume).

        Returns a ``_SyncFuture`` with a composite success/message dict.
        """
        results: list[dict] = []
        for i, (name, ref) in enumerate(self._replicas.items()):
            adjusted = self._adjust_payload(ep, payload, i)
            result = ref.handle_request.call_one(ep, adjusted).get(timeout=300)
            results.append(result)

        success = all(r.get("success", True) for r in results)
        message = "\n".join(r.get("message", "") for r in results)
        return _SyncFuture({"success": success, "message": message})

    # ------------------------------------------------------------------
    # Payload adjustment
    # ------------------------------------------------------------------

    def _adjust_payload(self, ep: str, payload: dict, replica_index: int) -> dict:
        """Adjust ``rank_offset`` for ``init_weights_update_group`` per replica.

        Each replica's vLLM workers need unique rank offsets within the
        shared HCCL group.  For replica *i*, the base offset is shifted
        by ``i * per_replica_workers`` where ``per_replica_workers`` is
        ``tp_size * pp_size`` (the number of HCCL participants per
        replica, **excluding** data-parallel workers).
        """
        if ep != "/areal_init_weights_update_group":
            return payload
        if len(self._placements) <= 1:
            return payload

        adjusted = dict(payload)
        original_offset = payload.get("rank_offset", 0)
        adjusted["rank_offset"] = (
            original_offset + replica_index * self._per_replica_workers
        )
        return adjusted
