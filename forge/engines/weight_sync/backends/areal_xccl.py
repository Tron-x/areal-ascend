"""AReaL-style xccl backend (placeholder, not implemented).

Fallback backend for environments where HiXL / torchstore isn't available
or trustworthy: use a second ``torch.distributed`` process group spanning
trainer rank 0 + every vLLM worker, have trainer rank 0 ``dist.broadcast``
gathered full tensors, have each worker receive into an empty buffer and
then call ``model.load_weights``.  Exactly the path AReaL itself uses (see
``/root/AReaL/areal/engine/fsdp_engine.py::_update_weights_from_distributed``
and ``/root/AReaL/areal/engine/vllm_ext/vllm_worker_extension.py::update_weight_xccl``).

Why keep this option even though multi-vol is faster:

* Zero HiXL dependency.  If CANN / HiXL has an issue on a given cluster,
  the xccl path still works with stock HCCL.

* Proven at scale.  AReaL has production traces running this pattern; we
  do not (yet).

* Debugging baseline.  When comparing our HiXL-based backends to a
  known-good reference, being able to flip a config flag and get the same
  weights delivered via HCCL broadcast is priceless.

AReaL's rank layout (preserved so our planner stays compatible):

    xccl_world = gen.world_size + 1   (the +1 is trainer rank 0)
    gen_worker rank = 1 + server_idx * tp_size * pp_size + local_rank
    trainer rank     = 0 (broadcast source)

Control plane differences from AReaL:

* AReaL uses HTTP to kick the inference side (``/init_weights_update_group``
  + ``/update_weights_from_distributed``).  We replace that with a Monarch
  actor endpoint call (``generator.workers.xccl_init_and_recv``) -- same
  semantics, different transport for metadata.

* AReaL's init sequence has trainer rank 0 call ``init_custom_process_group``
  with ``rank=0`` while inference workers call it with their own rank
  computed from ``rank_offset``.  We re-implement this directly since the
  AReaL function is internal to ``FSDPEngine``.

Rough API skeleton for when we implement:

    async def initialize(trainer, generator, layout, config):
        # Pick a free port on the trainer host.
        master_port = find_free_port(trainer_host)
        world = 1 + layout.gen_world
        # Parallel init on both sides.
        await asyncio.gather(
            trainer.xccl_init_group.call_one(
                master_addr=trainer_host, master_port=master_port,
                world_size=world, rank=0,
            ),
            generator.workers.xccl_init_group.fanout(
                master_addr=trainer_host, master_port=master_port,
                world_size=world, rank_offset=1,
            ),
        )
        self._pg_ready = True

    async def push(trainer, version):
        # Trainer rank 0 gathers state_dict and broadcasts each tensor on
        # the xccl group.  Driver hears back a summary.
        return await trainer.xccl_broadcast_all.call_one(version=version)

    async def pull(generator, version):
        # Workers are already recv-looping after the push call kicked them
        # (AReaL uses HTTP setup -> broadcast -> load loop).  Pull is a
        # no-op on this backend, or it waits for the worker-side load to
        # finish.
        return await generator.workers.xccl_wait_and_load.fanout(version=version)

Known limits (also in AReaL's docs):

* vLLM LoRA + xccl is unsupported (see
  ``areal/engine/sglang_remote.py`` XCCL LoRA rejection).
* ``gen.pp_size != 1`` raises NotImplementedError in AReaL's SGLang path.
* Rank 0 is the broadcast source, so its single NIC caps bandwidth -- no
  multi-NIC aggregation, unlike multi-vol.

Bandwidth comparison at 2-machine / 4-NPU-trainer scale:

    multi-vol:  put = 4 NICs parallel, pull = M NICs parallel
    xccl:       put = 1 NIC (rank 0 out) = one-shot NCCL tree broadcast
                pull = embedded in broadcast recv

So xccl is strictly slower for the put side but simpler; fallback value,
not primary.
"""

from __future__ import annotations

from typing import Any

from forge.engines.weight_sync.service import (
    ParallelLayout,
    PullResult,
    PushResult,
)


class ArealXcclBackend:
    """Placeholder -- raises immediately so callers get a clear message.

    See module docstring for the port-of-AReaL plan.
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "ArealXcclBackend is a Phase-2 placeholder.  "
            "It will wrap AReaL's FSDPEngine._update_weights_from_distributed "
            "path behind our WeightSyncBackend protocol to offer a "
            "HiXL-free fallback.  Use backend='torchstore_multi_vol' for now."
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
