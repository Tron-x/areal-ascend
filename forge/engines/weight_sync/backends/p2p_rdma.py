"""Peer-to-peer RDMA backend (placeholder, not implemented).

Skips the storage volume entirely: trainer rank 0 (or a coordinator per
TP group) allocates a single 2 MiB-aligned NPU flat buffer, wraps it in a
``monarch.rdma.RDMABuffer``, and publishes the resulting
``RdmaRemoteBuffer`` handle (which serializes the ``HixlBuffer`` engine_id
+ addr + size; see ``/root/monarch/monarch_rdma/src/rdma_components.rs``
lines 52-59) to the generator actor over Monarch actor RPC.

The generator fans the handle out to each vLLM worker; each worker
``read_into(local_model_param_flat)`` directly from the trainer's NPU
memory -- zero storage mediator, fully peer-to-peer.  With N trainer ranks
and M workers, this is N*M concurrent RDMA transfers saturating up to
N*M NICs (subject to CANN HCCL port-range exhaustion).

What's missing that prevents "just drop it in":

* **Handle lifecycle**.  RDMABuffer is tied to the trainer proc's ACL
  context; if trainer tears down between push and pull, the handle is
  dangling.  Need an explicit ref-count or a "worker ACKs back to trainer"
  handshake before the trainer can release the buffer.

* **Ping-pong slots**.  Two flat buffers (v0 / v1) alternating; each needs
  its own ``RDMABuffer`` registered up front.  Sequence: v0 register at
  init, put v0, pull v0, keep v0 registered for next time we need slot 0.

* **Fused param handling**.  Unlike the multi-vol backend where torchstore
  splits keys per-param, a single flat buffer carries everything
  concatenated.  The pull side needs the same layout plan the standalone
  harness ``forge/scripts/test_weight_sync_2node.py`` already computes
  (``_plan_layout`` / ``_chunk_offsets``); factor those out of the script
  and reuse.

* **TP-aware sharding**.  With TP>1 each worker wants only its shard; we
  either advertise N*tp distinct handles (each trainer exposes tp_size
  pre-sharded buffers) or the workers slice on read (wasteful with
  tp_size-1 redundant bytes).  Decision deferred.

Rough API skeleton for when we implement:

    async def initialize(trainer, generator, layout, config):
        # Ask trainer to alloc + register flat buf for each slot (v0, v1).
        handles = await trainer.alloc_p2p_rdma_slots.call(slots=2)
        self._handles = handles  # {slot: RdmaRemoteBuffer}
        # Tell generator workers to alloc matching flat buf + plan each.
        await generator.workers.prepare_p2p_rdma.fanout(plan)

    async def push(trainer, version):
        # Trainer rank 0: pack state_dict -> flat buffer (slot = v % 2).
        return await trainer.publish_to_p2p_slot.call(version=version)

    async def pull(generator, version):
        slot = version % 2
        handle = self._handles[slot]
        # Every worker reads the same handle into its own flat buffer, then
        # slices into model params.
        return await generator.workers.pull_from_p2p_handle.fanout(
            handle=handle, version=version
        )

See ``monarch/tests/hixl/e2e/test_multinode_rdma.py`` for the canonical
P2P handshake shape.
"""

from __future__ import annotations

from typing import Any

from forge.engines.weight_sync.service import (
    ParallelLayout,
    PullResult,
    PushResult,
)


class P2PRdmaBackend:
    """Placeholder -- raises immediately so callers get a clear message
    instead of a mysterious hang.  See module docstring for the
    implementation plan.
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "P2PRdmaBackend is a Phase-2 placeholder.  "
            "See forge/engines/weight_sync/backends/p2p_rdma.py docstring "
            "for the design sketch.  Use backend='torchstore_multi_vol' "
            "for now."
        )

    async def initialize(
        self,
        trainer_actor: Any,
        generator_actor: Any,
        layout: ParallelLayout,
        config: dict,
    ) -> dict:
        raise NotImplementedError

    async def push(self, trainer_actor: Any, version: int) -> PushResult:
        raise NotImplementedError

    async def pull(self, generator_actor: Any, version: int) -> PullResult:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError
