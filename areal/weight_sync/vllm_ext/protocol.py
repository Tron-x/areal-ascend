"""HTTP wire-protocol schemas for vLLM-side weight synchronisation.

Single source of truth for the request bodies the trainer-side
``WeightSyncClient`` posts and that the server-side router (or any
adapter shim, e.g. ms-swift's ``SwiftRolloutDeploy``) consumes.

Why a dedicated module
======================

Pre-extraction the same schemas were defined twice -- once in
``server_router.py`` (extending vLLM's ``OpenAIBaseModel``) and once
inside ``forge/engines/msswift/glue.py`` (as private ``_*Request``
classes mirroring the wire shape).  Any field added on one side and
forgotten on the other would silently desync the protocol -- tight,
implicit coupling between two files that have no compile-time link.

Centralising here means:

* L2 owns the wire format definitively (forge adapters import, never
  re-declare).
* Adding a field is a single PR diff against this file.
* New backends (sgang shim, llamafactory rollout, etc.) get the same
  schemas for free -- no per-adapter copy-paste.

Coupling boundary
=================

This module **only** depends on Pydantic.  No vLLM, no AReaL engine,
no Forge.  That keeps the schemas usable by any HTTP server / client
in the project regardless of which framework is installed.

The ``model_config = {"extra": "ignore"}`` setting means callers can
include extra fields for forward-compatibility without breaking older
servers -- crucial when one side of the wire upgrades first.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _WireBase(BaseModel):
    """Base for all wire-protocol schemas.

    ``extra='ignore'`` lets newer clients send fields that older
    servers don't know about (forward-compatible extension), while
    Pydantic still validates the fields we *do* declare.
    """

    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------- group setup


class UpdateGroupRequest(_WireBase):
    """Body of ``/areal_init_weights_update_group``.

    Mirrors the args ``init_custom_process_group`` needs to bind a
    cross-mesh torch.distributed group on each vLLM worker.

    Attributes:
        master_address: TCPStore master IP (trainer rank 0 binds, vLLM
            workers connect).  Reachable from every vLLM worker host.
        master_port: TCPStore master port (decimal string, not int --
            keeps wire shape stable across language clients).
        rank_offset: First vLLM rank in the group, 1-indexed past the
            trainer.  For trainer + N-way TP: trainer is rank 0 so
            ``rank_offset = 1``.
        world_size: Total ranks in the group (trainer + all vLLM).
        backend: ``"hccl"`` on Ascend, ``"nccl"`` on CUDA, ``"gloo"`` for
            CPU smoke tests.  Pass ``None`` to let the worker auto-detect.
        group_name: Human-readable identifier; vLLM keys
            ``weight_update_groups[group_name]`` by it so multiple
            independent sync channels can coexist (e.g. one per data-
            parallel rollout server).
    """

    master_address: str
    master_port: str
    rank_offset: int
    world_size: int
    backend: str
    group_name: str


# --------------------------------------------------------------- full sync


class UpdateWeightsFromXcclRequest(_WireBase):
    """Body of ``/areal_set_update_weight_meta``.

    Tells each vLLM worker the (name, dtype, shape) tuples it should
    expect to receive over XCCL via ``dist.broadcast`` in the
    *immediately following* ``/areal_update_weights_xccl`` round.

    The order of ``names`` MUST match the broadcast order on the
    trainer side -- workers iterate this list and call
    ``torch.distributed.broadcast`` per entry, so any reorder will
    silently load the wrong tensor into the wrong parameter slot.

    Attributes:
        names: Parameter names (in vLLM's runtime layout, after
            ``hf_to_vllm_mapper``); order = broadcast order.
        dtypes: Per-tensor dtype strings (``"bfloat16"`` /
            ``"torch.bfloat16"`` both accepted).
        shapes: Per-tensor shape lists.
        group_name: Which ``weight_update_groups`` entry to use; must
            match what was passed to ``/areal_init_weights_update_group``.
    """

    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str


class UpdateWeightsRequest(_WireBase):
    """Body of ``/areal_update_weights`` (disk-based, NOT XCCL).

    Used as a fallback path when the XCCL group can't be set up (e.g.
    backends without RDMA) -- the trainer dumps a checkpoint to a
    shared filesystem and the workers reload from there.

    Attributes:
        model_path: Filesystem path readable from every vLLM worker.
        load_format: vLLM ``load_format`` (e.g. ``"auto"``,
            ``"safetensors"``).  ``None`` = auto-detect.
        abort_all_requests: When True, vLLM cancels every inflight
            request before reload (prevents stale-token batches).
    """

    model_path: str
    load_format: str | None = "auto"
    abort_all_requests: bool = False


# --------------------------------------------------------------- LoRA sync


class UpdateWeightsRequestLora(_WireBase):
    """Body of ``/areal_update_weights_lora`` (disk-based LoRA).

    Disk-based counterpart to ``UpdateWeightsRequestLora`` -- same
    fallback role as ``UpdateWeightsRequest`` but tagged with the LoRA
    addressing tuple so the worker mounts via ``LoRARequest`` instead
    of replacing base weights.

    Attributes:
        lora_model_path: Filesystem path to the saved adapter.
        lora_name: Logical name vLLM will surface for routing requests.
        lora_int_id: Integer slot id; reusing the same id replaces the
            adapter in place.
        base_model_name: HF id / path of the base model the adapter
            was trained against (compatibility check).
        load_format / abort_all_requests: Same as the non-LoRA variant.
    """

    lora_model_path: str
    lora_name: str
    lora_int_id: int
    base_model_name: str
    load_format: str | None = "auto"
    abort_all_requests: bool = False


class UpdateWeightsFromXcclRequestLora(_WireBase):
    """Body of ``/areal_set_update_weight_meta_lora``.

    LoRA-incremental counterpart to
    :class:`UpdateWeightsFromXcclRequest`.  Carries the same
    ``names / dtypes / shapes`` (now describing the LoRA matrices,
    typically ``base_model.model.layers.<i>.self_attn.q_proj.lora_A.weight``
    etc.) plus the seven LoRA-addressing fields the worker needs to
    rebuild the adapter via ``LoRAModel.from_lora_tensors``:

    Attributes:
        names / dtypes / shapes / group_name: Same semantics as the
            full-broadcast variant; describe the LoRA delta tensors.
        lora_name: Logical adapter name (must match what
            ``LoRARequest`` / ``add_lora`` would use).
        lora_int_id: Integer slot id; first push *creates* the slot
            (worker handles missing-id case), subsequent pushes reuse.
        lora_target_modules: ``list[str]`` of layer names or a sentinel
            string like ``"all-linear"``.  Forwarded verbatim to
            ``PEFTHelper.from_dict``.
        lora_rank: PEFT ``r``.
        lora_alpha: PEFT ``alpha``.
        lora_bias: One of ``"none"`` / ``"all"`` / ``"lora_only"``.
        base_model_name: HF id / path of the base model.
    """

    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str
    lora_name: str
    lora_int_id: int
    lora_target_modules: list[str] | str
    lora_rank: int
    lora_alpha: int
    lora_bias: str
    base_model_name: str


__all__ = [
    "UpdateGroupRequest",
    "UpdateWeightsRequest",
    "UpdateWeightsRequestLora",
    "UpdateWeightsFromXcclRequest",
    "UpdateWeightsFromXcclRequestLora",
]
