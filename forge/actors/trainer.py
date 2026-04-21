"""TrainerActor — framework-agnostic training actor for Monarch orchestration.

Supports two backend protocols:

- **TrainEngine** (new): Framework-agnostic interface with external batch input
  and explicit weight sync separation. Use for FSDP, Megatron, etc.
- **TrainBackend** (legacy): AReaL-specific interface with internal data fetch,
  rollout+train coupling, and built-in weight sync.

The actor auto-detects which protocol to use based on what is passed at
construction time.

Orchestration modes (both protocols):

- **Synchronous**: ``train_step(step)`` (legacy) or ``train_on_engine_batch(batch, step)`` (new).
- **Async pipeline**: ``train_on_buffered_batch`` + ``sync_weights`` (legacy) or
  ``train_on_engine_batch`` with external ``WeightSyncStrategy`` (new).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

if TYPE_CHECKING:
    from forge.core.protocols import TrainBackend, TrainEngine

logger = logging.getLogger("TrainerActor")


class TrainerActor(ForgeActor):
    """Training actor with dual-protocol support (TrainEngine + TrainBackend).

    New code should pass ``engine=`` (a ``TrainEngine``). Legacy code
    continues to work by passing ``backend=`` (a ``TrainBackend``).

    New path (TrainEngine)::

        info = await actor.initialize.call()
        result = await actor.train_on_engine_batch.call(batch, step)
        spec = await actor.get_weights_spec.call()
        sd = await actor.state_dict_for_sync.call()

    Legacy path (TrainBackend)::

        info = await actor.initialize.call()
        result = await actor.train_on_buffered_batch.call(batch, step)
        await actor.sync_weights.call(step)
    """

    procs = 1
    with_gpus = True

    def __init__(
        self,
        backend: TrainBackend | None = None,
        engine: TrainEngine | None = None,
        **kwargs: Any,
    ):
        if backend is None and engine is None:
            raise ValueError("Either 'backend' or 'engine' must be provided")
        self._backend = backend
        self._engine = engine
        self._use_engine = engine is not None
        self._extra_kwargs = kwargs

    # ==================================================================
    # Shared endpoints
    # ==================================================================

    @endpoint
    def initialize(self) -> dict:
        """Initialize the training backend/engine and return metadata."""
        if self._use_engine:
            return self._engine.initialize()
        return self._backend.initialize()

    @endpoint
    def shutdown(self) -> None:
        """Shut down the training backend/engine."""
        if self._use_engine:
            self._engine.shutdown()
        else:
            self._backend.shutdown()

    # ==================================================================
    # New TrainEngine endpoints
    # ==================================================================

    @endpoint
    def train_on_engine_batch(self, batch: dict, step: int) -> dict:
        """Train on an externally-provided batch (TrainEngine path).

        The batch should already be adapted by a ``BatchAdapter``.
        """
        if not self._use_engine:
            raise RuntimeError(
                "train_on_engine_batch requires a TrainEngine. "
                "This actor was constructed with a legacy TrainBackend."
            )
        return self._engine.train_step(batch, step)

    @endpoint
    def get_weights_spec(self) -> dict:
        """Return the WeightsSpec as a dict (TrainEngine path).

        Used by WeightSyncStrategy to negotiate sync channels.
        """
        if not self._use_engine:
            raise RuntimeError("get_weights_spec requires a TrainEngine")
        from dataclasses import asdict

        return asdict(self._engine.get_weights_spec())

    @endpoint
    def state_dict_for_sync(self) -> dict:
        """Return the model state dict for weight sync (TrainEngine path).

        The WeightSyncStrategy uses this to transfer weights to Generator.
        """
        if not self._use_engine:
            raise RuntimeError("state_dict_for_sync requires a TrainEngine")
        return self._engine.state_dict_for_sync()

    @endpoint
    def push_weights_torchstore(self, policy_version: int) -> dict:
        """Push the engine's state_dict to torchstore for TorchstoreWeightSync.

        Every parameter is stored under ``policy_ver_{v%2:010d}.{name}``
        (see ``forge.engines.weight_sync.torchstore_sync.get_param_key``);
        the ``{0|1}`` slot ping-pong keeps the store bounded by two
        versions, matching torchforge's scheme.

        ``ts.put_batch(entries)`` hands the whole state_dict to
        torchstore in a single transport-buffer setup -- this is the
        same API upstream torchforge uses in
        ``src/forge/actors/trainer/titan.py::push_weights``.  An earlier
        version of this method drove a per-parameter ``for ... ts.put``
        loop which, on CANN/HiXL, amplified every per-registration
        cost into 300 back-to-back HiXL ``RegisterBoundMems`` calls and
        eventually starved the RA HDC driver (``ra_hdc_typical_mr
        ret=-13``).

        An even more aggressive "pack into one flat 2 MiB-aligned NPU
        buffer, put one chunk at a time" variant was evaluated and
        regressed end-to-end sync latency from ~3 s to ~12 s for
        Qwen3-0.6B: the extra per-parameter ``full_tensor()`` gather
        collectives + pack copies on the trainer, combined with the
        per-view ``.cpu()`` copies on the generator side, dominate any
        win from the reduced HiXL registration count at this model
        size.  If that balance shifts at larger models, revisit the
        flat-buffer path using ``forge.engines.weight_sync._flat_layout``.

        Only rank 0 publishes; other ranks participate in the FSDP
        gather inside ``state_dict_for_sync`` and then exit early.
        """
        if not self._use_engine:
            raise RuntimeError("push_weights_torchstore requires a TrainEngine")
        import asyncio
        import time

        import torch.distributed as dist
        import torchstore as ts

        from forge.engines.weight_sync.torchstore_sync import get_param_key

        # All ranks participate: FSDP/HSDP sharded tensors need the
        # collective gather to materialise full tensors (and
        # ``state_dict_for_sync`` also runs ``DTensor.full_tensor()`` +
        # ``sd_adapter.to_hf`` underneath, both of which need all
        # ranks).
        build_t0 = time.perf_counter()
        state_dict = self._engine.state_dict_for_sync()
        build_s = time.perf_counter() - build_t0

        rank = dist.get_rank() if dist.is_initialized() else 0

        total_bytes = 0
        param_names: list[str] = []
        param_shapes: list[tuple] = []
        param_dtypes: list[str] = []
        for name, tensor in state_dict.items():
            param_names.append(name)
            param_shapes.append(tuple(tensor.shape))
            param_dtypes.append(str(tensor.dtype).replace("torch.", ""))
            total_bytes += tensor.numel() * tensor.element_size()

        async def _do_puts() -> float:
            put_t0 = time.perf_counter()
            entries = {
                get_param_key(policy_version, name_): tensor_
                for name_, tensor_ in state_dict.items()
            }
            await ts.put_batch(entries)
            return time.perf_counter() - put_t0

        put_s = asyncio.run(_do_puts()) if rank == 0 else 0.0

        return {
            "num_keys": len(param_names),
            "bytes": total_bytes,
            "build_state_dict_s": build_s,
            "put_s": put_s,
            "param_names": param_names,
            "param_shapes": param_shapes,
            "param_dtypes": param_dtypes,
            "rank": rank,
        }

    @endpoint
    def get_engine_metadata(self) -> dict:
        """Return engine metadata (TrainEngine path)."""
        if not self._use_engine:
            raise RuntimeError("get_engine_metadata requires a TrainEngine")
        return self._engine.get_metadata()

    # ==================================================================
    # Legacy TrainBackend endpoints
    # ==================================================================

    @endpoint
    def train_step(self, global_step: int) -> dict:
        """Combined rollout + training in one step (legacy synchronous mode)."""
        if self._use_engine:
            raise RuntimeError(
                "train_step(global_step) is the legacy synchronous path. "
                "Use train_on_engine_batch(batch, step) with TrainEngine."
            )
        return self._backend.train_step(global_step)

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        """Run rollout only, return serialized batch (legacy)."""
        if self._use_engine:
            raise RuntimeError("do_rollout is only available with TrainBackend")
        return self._backend.do_rollout(global_step)

    @endpoint
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        """Train on a pre-produced rollout batch (legacy)."""
        if self._use_engine:
            return self._engine.train_step(batch_data, global_step)
        return self._backend.train_on_batch(batch_data, global_step)

    @endpoint
    def train_on_buffered_batch(
        self,
        batch_data: dict,
        global_step: int,
        skip_weight_sync: bool = False,
    ) -> dict:
        """Train on a batch from ReplayBuffer (legacy async pipeline mode).

        When ``skip_weight_sync=True``, call ``sync_weights`` separately
        to push updated weights to Generator.
        """
        if self._use_engine:
            return self._engine.train_step(batch_data, global_step)
        return self._backend.train_on_buffered_batch(
            batch_data, global_step, skip_weight_sync=skip_weight_sync
        )

    @endpoint
    def sync_weights(self, global_step: int) -> dict:
        """Push updated weights to Generator (legacy async pipeline mode)."""
        if self._use_engine:
            raise RuntimeError(
                "sync_weights is a legacy endpoint. "
                "Use WeightSyncStrategy externally with TrainEngine."
            )
        return self._backend.sync_weights(global_step)

    @endpoint
    def get_train_metadata(self) -> dict:
        """Return training metadata for the orchestrator."""
        if self._use_engine:
            return self._engine.get_metadata()
        return self._backend.get_train_metadata()
