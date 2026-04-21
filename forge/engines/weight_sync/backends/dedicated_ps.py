"""Dedicated parameter-server backend (placeholder, not implemented).

Like ``torchstore_multi_vol`` in API and key scheme, but the ``K`` storage
volumes live on a separate ``layout.ps_mesh_name`` host mesh rather than
colocated with the trainer.

Deployment flavors (all behave identically from the caller's perspective;
the difference is the ``hostfile`` label):

* **Dedicated PS machine(s)**:  3rd+ node, hostfile entry
  ``192.168.0.99:0-7 label=ps`` -- ``K=8`` PS procs on that machine.
  The only topology that truly isolates PS traffic from trainer/generator
  NIC sharing.

* **Colocated PS on trainer host's free NPUs**:  two-machine setup, hostfile
  ``192.168.0.23:4-7 label=ps`` -- ``K=4`` PS procs on node 23 NPUs 4-7
  (trainer uses 0-3).  Put path: intra-host HCCS.  Pull path: cross-node
  RoCE.

* **Colocated PS on generator host's free NPUs**:  ``192.168.0.26:1-4
  label=ps`` -- K=4 PS procs on the generator host but different NPUs than
  vLLM.  Put path: cross-node RoCE.  Pull path: intra-host HCCS.

* **Split PS**:  ``192.168.0.23:4-5 label=ps, 192.168.0.26:4-5 label=ps``
  -- 2+2 PS, half cross-node for put *and* pull, most balanced.

What's missing to ship:

1. ``BareMetalLauncher`` extension to parse ``label=ps`` with an NPU
   *range* (currently it only handles whole-host slot counts).  Minimal
   form: the label defines a new named host mesh that
   ``provisioner.get_host_mesh("ps")`` can return.

2. ``TopologyPlanner`` knows how to hash-route parameter names to K PS
   ranks (``ps_rank = hash(name) % K``) if we want the K PS ranks to
   each hold only ``1/K`` of the state_dict (saves storage by ``K``).
   For Phase-1 interop we can punt: every PS rank holds the full
   state_dict (same redundancy as ``torchstore_multi_vol``) so the key
   scheme stays identical and only the mesh changes.

3. Trainer-side rank routing: rank i writes only the keys assigned to it
   by the planner (if we do hash-route); or rank i writes to some subset
   of PS ranks (if we do N:K fan-out).  Both are planner extensions.

Rough API skeleton for when we implement:

    async def initialize(trainer, generator, layout, config):
        provisioner = await _get_provisioner()
        ps_hosts = await provisioner.get_host_mesh(layout.ps_mesh_name)
        K = layout.ps_world or ps_hosts.size()
        self._storage_mesh = ps_hosts.spawn_procs(
            per_host={"procs": K // ps_hosts.size()},
            bootstrap=storage_bootstrap_factory(npu_base=0),
        )
        await ts.initialize(num_storage_volumes=K, strategy=..., mesh=...)

    async def push / pull: identical to MultiVolTorchstoreBackend.

The Service / Trainer / Generator code is agnostic to the difference;
that's the whole point of the backend abstraction.
"""

from __future__ import annotations

from typing import Any

from forge.engines.weight_sync.service import (
    ParallelLayout,
    PullResult,
    PushResult,
)


class DedicatedPsBackend:
    """Placeholder -- raises immediately so callers get a clear message.

    See module docstring for the design sketch and deployment flavors.
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "DedicatedPsBackend is a Phase-2 placeholder.  "
            "Needs BareMetalLauncher label=ps support first.  "
            "See forge/engines/weight_sync/backends/dedicated_ps.py "
            "docstring for the plan."
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
