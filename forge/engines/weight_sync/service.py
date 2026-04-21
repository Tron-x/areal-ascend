"""Weight-sync Service: clean control / data plane separation.

This module is the **control plane**.  It owns the orchestration of a single
weight-sync cycle -- pause generation, publish metadata, run the data transfer
via a backend, resume generation -- and keeps **zero tensor bytes in-process**.
The actual bytes flow through whatever ``WeightSyncBackend`` the caller picks
(torchstore multi-volume, P2P RDMA, dedicated PS node, AReaL-style HCCL
broadcast, ...).

Layering

    L1  Driver (``forge/apps/grpo.py``)
            |
            v  service.push_and_pull(version)
    L2  WeightSyncService                  <-- THIS FILE
            |
            +--> TopologyPlanner       (logical parallel layout -> rank plan)
            |
            +--> WeightSyncBackend     (pluggable, data plane lives here)
            |
            +--> TrainerActor.publish_weights      (actor RPC, metadata only)
            |
            +--> Generator.workers.pull_weights    (actor RPC, fanout)

Design invariants worth enforcing as the backend set grows:

1. **Service never touches tensors**.  Everything size-O(model) happens inside
   actor endpoints (``publish_weights`` on trainer ranks, ``pull_weights`` on
   vLLM workers).  The Service orchestrates via Monarch RPC carrying only
   metadata (version, layout, keys).

2. **Backend API is scale-invariant**.  The same ``push / pull`` signatures
   must work for 2-node/8-NPU and 32-node/256-NPU topologies.  Growth is
   handled inside the backend (more volumes, K dedicated PS nodes, shard-aware
   P2P handles) and/or inside ``TopologyPlanner`` (mapping logical ranks to
   physical rank layouts).

3. **Control plane is O(1) in worker count**.  ``generator.workers.pull_weights
   .fanout(plan)`` is a single fanout call, not a loop.  ``PullPlan`` encodes
   per-worker slices compactly so its payload stays bounded regardless of
   model size or worker count (no ``list[311 * N_workers]`` explosion).

The four-beat sync cycle (borrowed from AReaL's xccl orchestration:
pause -> set-meta -> transfer -> resume) is what makes any future backend
swap transparent to the caller -- the beats stay, the bytes path changes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("WeightSyncService")


# ============================================================================
# Metadata / plan dataclasses
# ============================================================================


@dataclass
class ParamMeta:
    """Per-parameter description produced by the trainer after state-dict gather.

    All fields are JSON-friendly (no tensor refs) so a ``list[ParamMeta]``
    can ride over Monarch RPC cheaply.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str  # torch dtype with "torch." stripped, e.g. "bfloat16"
    nbytes: int


@dataclass
class TensorSliceSpec:
    """A (dim, start, stop) slice of a full-shape parameter.

    Used by TP-aware backends to let a worker request only the shard it
    actually needs instead of pulling the full tensor and discarding 1 / tp_size.
    Phase-1 leaves this as ``None`` (TP=1 case): worker pulls the full tensor.
    """

    dim: int
    start: int
    stop: int


@dataclass
class PullItem:
    """One (key, optional slice) pair a worker will fetch."""

    name: str
    key: str  # backend-computed storage key, e.g. "policy_ver_0000000000.embed"
    slice_spec: TensorSliceSpec | None = None


@dataclass
class PullPlan:
    """Per-worker pull schedule.

    ``per_worker[worker_rank]`` is the list of items that the vLLM worker at
    ``worker_rank`` should ``ts.get(inplace_tensor=self.model_param[name])``.
    For TP=1 this is the same item list on every worker; for TP>1 each rank
    gets its own shard via ``slice_spec``.
    """

    version: int
    per_worker: dict[int, list[PullItem]]
    backend_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PushPlan:
    """Per-trainer-rank publish schedule.

    For torchstore multi-volume: every rank publishes the full gathered
    state_dict to its local volume (LocalRankStrategy routes by rank).
    For P2P RDMA: only rank 0 publishes a flat buffer handle.
    For dedicated PS: rank i sends to one of K PS nodes selected by hash.
    """

    version: int
    keys_per_rank: dict[int, list[str]]  # trainer_rank -> keys it writes
    backend_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PushResult:
    """Returned by ``WeightSyncBackend.push``."""

    version: int
    num_keys: int
    bytes_total: int
    push_s: float
    build_state_dict_s: float = 0.0
    per_rank: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class PullResult:
    """Returned by ``WeightSyncBackend.pull``."""

    version: int
    num_keys: int
    bytes_total: int
    pull_s: float
    workers_load_s: float = 0.0
    per_worker: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SyncSummary:
    """Aggregate stats returned by ``WeightSyncService.push_and_pull``."""

    version: int
    success: bool
    num_keys: int
    bytes_total: int
    push_s: float
    pull_s: float
    total_s: float
    push_gbps: float = 0.0
    pull_gbps: float = 0.0


# ============================================================================
# Parallel layout + topology planner
# ============================================================================


@dataclass
class ParallelLayout:
    """Logical parallelism shape -- never references physical NPUs.

    The backend consumes this to decide storage topology, rank fan-in/out, and
    for future TP-aware sharding.  Keeping the API on the logical layout means
    the planner / backend is the same code path at 8 NPUs and 256 NPUs.

    ``ps_world == 0`` is the sentinel for "let the backend pick".  For
    ``torchstore_multi_vol`` that means ps_world := train_world (colocate N
    volumes with N trainer ranks, no separate PS host).  For
    ``dedicated_ps`` it means "read ps_world from the ps host mesh size".
    """

    train_world: int
    gen_world: int
    gen_tp: int = 1
    gen_pp: int = 1
    ps_world: int = 0  # 0 = backend-decides
    trainer_mesh_name: str = "trainer"
    generator_mesh_name: str = "generator"
    ps_mesh_name: str = "ps"  # only used by backends that dedicate a PS host mesh


class TopologyPlanner:
    """Map logical ranks (from ``ParallelLayout``) to the rank layout a
    backend needs to run push/pull.

    Borrows the AReaL ``rank_offset = 1 + server_idx * tp * pp + local_rank``
    convention for consistency with the eventual ``areal_xccl`` backend.

    Phase-1 responsibility:

    * For each ``(param_name, shape, dtype)`` in the published layout, decide
      which worker(s) need what slice (full tensor when TP=1, ``TensorSliceSpec``
      otherwise).
    * Decide which storage key(s) each trainer rank writes (scheme is backend-
      specific but the planner can provide defaults).

    Phase-2+ responsibility (not implemented yet):

    * For a K-node dedicated PS layout, hash-route param names to PS ranks
      (e.g. by layer index) so push/pull stay balanced.
    * For P2P RDMA, produce an N-pair list mapping trainer_rank i to
      generator_rank j given a chosen pairing strategy.
    """

    def __init__(self, layout: ParallelLayout) -> None:
        self.layout = layout

    def build_pull_plan(
        self,
        version: int,
        param_meta: list[ParamMeta],
        key_for: Callable[[int, str], str],
    ) -> PullPlan:
        """Default: every generator worker pulls every param (TP=1).

        ``key_for(version, name) -> str`` is the backend's key scheme.  The
        planner doesn't hard-code the prefix so different backends can pick
        their own (torchstore uses ``policy_ver_{0|1}``, a future P2P backend
        might use handle IDs, etc.).
        """
        tp = self.layout.gen_tp
        pp = self.layout.gen_pp
        if tp != 1 or pp != 1:
            # Phase-2 will emit per-rank TensorSliceSpec here.
            raise NotImplementedError(
                "TopologyPlanner.build_pull_plan: TP>1 or PP>1 not yet "
                "supported.  Current layout: "
                f"gen_tp={tp}, gen_pp={pp}, gen_world={self.layout.gen_world}."
            )

        items = [
            PullItem(name=m.name, key=key_for(version, m.name), slice_spec=None)
            for m in param_meta
        ]
        per_worker = {r: items for r in range(self.layout.gen_world)}
        return PullPlan(version=version, per_worker=per_worker)

    def build_push_plan(
        self,
        version: int,
        param_meta: list[ParamMeta],
        key_for: Callable[[int, str], str],
    ) -> PushPlan:
        """Default: every trainer rank writes the same set of keys.

        torchstore's ``LocalRankStrategy`` routes each rank's writes to its
        colocated volume, so writing the "same keys" from each rank is fine --
        each goes to a different storage volume.  The per-volume fanout
        happens inside torchstore, not here.
        """
        keys = [key_for(version, m.name) for m in param_meta]
        keys_per_rank = {r: keys for r in range(self.layout.train_world)}
        return PushPlan(version=version, keys_per_rank=keys_per_rank)


# ============================================================================
# Backend protocol
# ============================================================================


@runtime_checkable
class WeightSyncBackend(Protocol):
    """Pluggable data-plane backend.

    Implementations live in ``forge.engines.weight_sync.backends.*``.  Four
    methods, all async, all operating with the current version number.

    Contract:

    * ``initialize`` is called once at startup.  Backend sets up storage
      volumes / PS procs / process groups / handle caches; holds a reference
      to the ``ParallelLayout`` to drive its own internal planning.
    * ``push`` / ``pull`` are called once per training step.  Must be
      idempotent under ping-pong slot reuse (version ``v`` and ``v+2`` share
      the same slot; the backend is expected to overwrite cleanly).
    * ``shutdown`` releases resources.  Should not raise in the common path
      since it's called from ``finally`` blocks.

    Plans (``PushPlan`` / ``PullPlan``) are backend-internal concerns -- the
    Service does not construct them, because different backends need very
    different plan shapes (multi-vol has no per-rank plan, P2P has a handle
    map, dedicated_ps has a hash-routed key map).  The dataclasses above are
    helpers that backends are free to use; they are not part of this contract.
    """

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        layout: ParallelLayout,
        config: dict,
    ) -> dict:
        """Wire up data-plane resources.  Returns backend metadata."""
        ...

    async def push(
        self,
        trainer_actor: Any,
        version: int,
    ) -> PushResult:
        """Drive trainer-side data transfer for ``version``.

        Typical implementation: call ``trainer.publish_weights.call(version,
        backend_hints)`` to fan out to all trainer ranks, wait for their
        combined PushResult-worthy stats, cache the param meta if the
        backend needs it for the subsequent pull call.
        """
        ...

    async def pull(
        self,
        generator_actor: Any,
        version: int,
    ) -> PullResult:
        """Drive generator-side data transfer for ``version``.

        Typical implementation: call a generator-side endpoint (either on
        the generator actor directly or fanned out to its vLLM workers)
        that does the ``ts.get`` / ``RDMABuffer.read_into`` / collective
        broadcast-recv and reports back aggregated stats.
        """
        ...

    async def shutdown(self) -> None:
        """Release storage volumes / process groups / handles."""
        ...


# ============================================================================
# Service: control-plane orchestration only
# ============================================================================


class WeightSyncService:
    """Control-plane orchestrator.

    A single cycle runs the four beats:

        pause (generator) -> publish (trainer) -> pull (gen workers) -> resume

    Each beat is a Monarch RPC carrying metadata only.  Tensor bytes never
    pass through this process.

    The Service is deliberately **not** a Monarch actor in Phase-1; it is a
    plain Python object held by the driver (``forge/apps/grpo.py``).  That
    avoids a needless extra proc hop for a layer that is already bound to the
    driver's asyncio loop.  If we ever want to address weight sync from a
    remote process we'll wrap this class in a ``ServiceActor`` following
    torchforge's pattern.
    """

    def __init__(
        self,
        backend: WeightSyncBackend,
        trainer_actor: Any,
        generator_actor: Any,
        layout: ParallelLayout,
        planner: TopologyPlanner | None = None,
        config: dict | None = None,
    ) -> None:
        self.backend = backend
        self.trainer = trainer_actor
        self.generator = generator_actor
        self.layout = layout
        self.planner = planner or TopologyPlanner(layout)
        self.config = config or {}
        self._initialized = False
        self._current_version = -1

    async def initialize(self) -> dict:
        meta = await self.backend.initialize(
            self.trainer, self.generator, self.layout, self.config
        )
        self._initialized = True
        logger.info(
            "WeightSyncService initialized (backend=%s, layout=train=%d, "
            "gen=%d, tp=%d, pp=%d)",
            type(self.backend).__name__,
            self.layout.train_world,
            self.layout.gen_world,
            self.layout.gen_tp,
            self.layout.gen_pp,
        )
        return meta

    async def push_and_pull(self, version: int) -> SyncSummary:
        """Run one full sync cycle.

        Two phases, both driven by the backend:

        1. ``backend.push(trainer, version)``:
           trainer ranks gather their state_dict and write to the backend's
           storage (torchstore volumes / RDMABuffer handles / etc.).  Backend
           caches whatever metadata it needs for the subsequent pull.

        2. ``backend.pull(generator, version)``:
           generator-side workers read directly from the backend's storage
           straight into their vLLM model params (inplace HiXL RDMA for
           torchstore multi-vol; broadcast-recv for xccl; etc.).

        Service just measures wall-clock and aggregates the stats -- all
        bytes flow through the backend path.
        """
        if not self._initialized:
            raise RuntimeError("WeightSyncService.push_and_pull: not initialized")

        total_t0 = time.perf_counter()

        push_t0 = time.perf_counter()
        push_result = await self.backend.push(self.trainer, version)
        push_s = time.perf_counter() - push_t0

        pull_t0 = time.perf_counter()
        pull_result = await self.backend.pull(self.generator, version)
        pull_s = time.perf_counter() - pull_t0

        total_s = time.perf_counter() - total_t0
        bytes_total = max(push_result.bytes_total, pull_result.bytes_total)
        success = pull_result.num_keys > 0

        if success:
            self._current_version = version

        summary = SyncSummary(
            version=version,
            success=success,
            num_keys=pull_result.num_keys,
            bytes_total=bytes_total,
            push_s=push_s,
            pull_s=pull_s,
            total_s=total_s,
            push_gbps=(bytes_total / push_s / (1024**3)) if push_s > 0 else 0.0,
            pull_gbps=(bytes_total / pull_s / (1024**3)) if pull_s > 0 else 0.0,
        )
        logger.info(
            "weight_sync v%d: %d keys, %.2f GB, push=%.2fs (%.2f GB/s), "
            "pull=%.2fs (%.2f GB/s), total=%.2fs",
            version,
            summary.num_keys,
            bytes_total / (1024**3),
            push_s,
            summary.push_gbps,
            pull_s,
            summary.pull_gbps,
            total_s,
        )
        return summary

    # ------------------------------------------------------------------ legacy
    # Compatibility alias: the old ``WeightSyncStrategy.push(version)`` signature
    # is what ``forge/apps/grpo.py`` currently calls.  Keep a thin adapter so
    # the driver loop can migrate gradually.
    async def push(self, version: int) -> dict:
        summary = await self.push_and_pull(version)
        return {
            "version": summary.version,
            "success": summary.success,
            "num_keys": summary.num_keys,
            "bytes": summary.bytes_total,
            "push_s": summary.push_s,
            "pull_s": summary.pull_s,
            "total_s": summary.total_s,
            "push_gbps": summary.push_gbps,
            "pull_gbps": summary.pull_gbps,
        }

    async def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "current_version": self._current_version,
            "backend": type(self.backend).__name__,
            "layout": {
                "train_world": self.layout.train_world,
                "gen_world": self.layout.gen_world,
                "gen_tp": self.layout.gen_tp,
                "gen_pp": self.layout.gen_pp,
                "ps_world": self.layout.ps_world,
            },
        }

    async def shutdown(self) -> None:
        if not self._initialized:
            return
        try:
            await self.backend.shutdown()
        except Exception:
            logger.exception("backend shutdown raised; continuing")
        self._initialized = False
        logger.info("WeightSyncService shutdown")
