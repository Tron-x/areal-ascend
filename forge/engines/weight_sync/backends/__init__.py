"""Pluggable weight-sync backends.

Each backend implements :class:`forge.engines.weight_sync.service.WeightSyncBackend`.
The registry below lets the driver pick one by string name, so configuration
stays YAML-friendly and swapping backends at deploy time is a one-line change.

Known names (match the ``FORGE_WEIGHT_SYNC_BACKEND`` env / config key):

* ``"torchstore_multi_vol"`` -- Phase-1 default.  N storage volumes colocated
  with the N trainer ranks; trainer rank i ``ts.put_batch`` routes to volume i
  via ``LocalRankStrategy``, vLLM workers ``ts.get(inplace_tensor=param)``
  directly into their model parameters (no CPU staging).  Multi-NIC bandwidth
  on the put side, single-flow on the pull side (per worker).

* ``"p2p_rdma"`` -- (placeholder) Direct ``RDMABuffer`` handle exchange
  between trainer and generator procs, no storage volume.  Highest theoretical
  bandwidth; requires custom handle lifecycle + ping-pong slot management.

* ``"dedicated_ps"`` -- (placeholder) K storage volumes on a dedicated host
  mesh (``layout.ps_mesh_name``) rather than colocated with trainer.  Lets
  users add a separate parameter-server node.  Same API as multi_vol.

* ``"areal_xccl"`` -- (placeholder) Adapter over AReaL's existing
  ``FSDPEngine._update_weights_from_distributed`` path (rank-0 HCCL broadcast
  on a second torch.distributed process group).  Safe fallback when HiXL /
  torchstore has issues; no RDMA required.

The registry is intentionally small and explicit -- backends are big enough
concerns that dynamic import-on-demand hurts more than it helps.
"""

from __future__ import annotations

from typing import Any

from forge.engines.weight_sync.service import WeightSyncBackend

_BACKEND_LOADERS: dict[str, Any] = {}


def _register(name: str):
    """Decorator to register a backend-loading function under a string name.

    The wrapped loader is called on first request (lazy) so importing this
    module does not force every backend's optional deps (torchstore, HCCL,
    RDMABuffer, ...) to be importable.
    """

    def wrap(fn):
        _BACKEND_LOADERS[name] = fn
        return fn

    return wrap


@_register("torchstore_multi_vol")
def _load_torchstore_multi_vol(**kwargs) -> WeightSyncBackend:
    from forge.engines.weight_sync.backends.torchstore_multi_vol import (
        MultiVolTorchstoreBackend,
    )

    return MultiVolTorchstoreBackend(**kwargs)


@_register("p2p_rdma")
def _load_p2p_rdma(**kwargs) -> WeightSyncBackend:
    from forge.engines.weight_sync.backends.p2p_rdma import P2PRdmaBackend

    return P2PRdmaBackend(**kwargs)


@_register("dedicated_ps")
def _load_dedicated_ps(**kwargs) -> WeightSyncBackend:
    from forge.engines.weight_sync.backends.dedicated_ps import DedicatedPsBackend

    return DedicatedPsBackend(**kwargs)


@_register("areal_xccl")
def _load_areal_xccl(**kwargs) -> WeightSyncBackend:
    from forge.engines.weight_sync.backends.areal_xccl import ArealXcclBackend

    return ArealXcclBackend(**kwargs)


@_register("collective_broadcast")
def _load_collective_broadcast(**kwargs) -> WeightSyncBackend:
    from forge.engines.weight_sync.backends.collective_broadcast import (
        CollectiveBroadcastBackend,
    )

    return CollectiveBroadcastBackend(**kwargs)


def create_backend(name: str, **kwargs) -> WeightSyncBackend:
    """Instantiate a backend by name.

    Raises ``ValueError`` on unknown names with a hint listing the registered
    options.  ``kwargs`` are forwarded verbatim to the backend constructor.
    """
    if name not in _BACKEND_LOADERS:
        raise ValueError(
            f"Unknown weight_sync backend {name!r}.  "
            f"Available: {sorted(_BACKEND_LOADERS.keys())}"
        )
    return _BACKEND_LOADERS[name](**kwargs)


def available_backends() -> list[str]:
    """Return the registered backend names, sorted."""
    return sorted(_BACKEND_LOADERS.keys())


__all__ = ["create_backend", "available_backends"]
