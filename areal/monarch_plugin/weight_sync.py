"""XCCL/HCCL weight-update ``alloc_mode`` planning for Monarch + AReaL.

Two layers (see also ``topology.py``):

1. **Device / process topology** — which devices run inference vs training
   (``allocation_mode`` can say ``vllm:d4p1t1+…`` meaning four inference **devices**).
2. **Weight sync (this module)** — shape ``WeightUpdateMeta.alloc_mode`` so FSDP and
   vLLM agree on the custom process group.

``GeneratorActor`` uses Monarch’s embedded vLLM. The number of ranks that call
``init_update_weight_group`` follows **vLLM’s resolved parallel config**
(``tensor_parallel_size × pipeline_parallel_size × data_parallel_size`` from
``vLLMConfig.build_args``), which may keep ``data_parallel_size=1`` even when
the cluster reserves four NPUs for inference. Using ``allocation_mode.gen`` alone
for XCCL can then over-count and hang HCCL.

**Default:** build the inference half from that **runtime** parallel strategy
(see ``build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime``).

**Override:** ``MONARCH_INFERENCE_XCCL_PARTICIPANTS=N`` forces a flat ``d{N}p1t1``
inference half.

**High-level entry point:** ``resolve_xccl_alloc_mode`` encapsulates the full
decision — env override, vLLM runtime extraction, and alloc_mode construction —
so callers (launcher) only need one call.
"""

from __future__ import annotations

import logging
import os

from areal.api.alloc_mode import (
    AllocationMode,
    InferenceParallelism,
    ParallelStrategy,
)

logger = logging.getLogger("MonarchWeightSync")


def optional_flat_inference_xccl_from_env() -> int | None:
    """If ``MONARCH_INFERENCE_XCCL_PARTICIPANTS`` is set, return it; else ``None``."""
    raw = os.environ.get("MONARCH_INFERENCE_XCCL_PARTICIPANTS")
    if raw is None or str(raw).strip() == "":
        return None
    v = int(raw)
    if v < 1:
        raise ValueError(
            f"MONARCH_INFERENCE_XCCL_PARTICIPANTS must be >= 1, got {v!r}"
        )
    return v


def monarch_xccl_weight_update_alloc_mode_flat(
    train_world_size: int,
    inference_dp: int,
    *,
    rollout_backend: str = "vllm",
) -> AllocationMode:
    """Force inference to ``d{inference_dp}p1t1`` (escape hatch; see module doc)."""
    if train_world_size < 1:
        raise ValueError(f"train_world_size must be >= 1, got {train_world_size}")
    if inference_dp < 1:
        raise ValueError(f"inference_dp must be >= 1, got {inference_dp}")
    s = (
        f"{rollout_backend}:d{inference_dp}p1t1+"
        f"d{train_world_size}p1t1"
    )
    return AllocationMode.from_str(s)


def build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
    vllm_gen_parallel: ParallelStrategy,
    *,
    gen_backend: str,
    train_world_size: int,
    flat_inference_dp: int | None = None,
) -> AllocationMode:
    """Build ``AllocationMode`` for XCCL from **embedded vLLM** parallel dims.

    Parameters
    ----------
    vllm_gen_parallel
        Typically from ``vLLMConfig.build_args`` (tensor/pipeline/data parallel
        sizes after merging YAML with ``allocation_mode`` tp/pp).
    gen_backend
        e.g. ``"vllm"``.
    train_world_size
        FSDP / training ProcMesh world size.
    flat_inference_dp
        If set, inference half is forced to ``d{flat_inference_dp}p1t1``.
    """
    if train_world_size < 1:
        raise ValueError(f"train_world_size must be >= 1, got {train_world_size}")

    bb = gen_backend or "vllm"
    if flat_inference_dp is not None:
        return monarch_xccl_weight_update_alloc_mode_flat(
            train_world_size,
            flat_inference_dp,
            rollout_backend=bb,
        )

    gen_str = str(InferenceParallelism(bb, vllm_gen_parallel))
    train_str = f"d{train_world_size}p1t1"
    return AllocationMode.from_str(f"{gen_str}+{train_str}")


def _vllm_gen_parallel(config, alloc_mode) -> ParallelStrategy:
    """Extract vLLM **runtime** parallel strategy for XCCL inference half."""
    from areal.api.cli_args import vLLMConfig  # noqa: avoid circular / heavy import at top

    args_dict = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=alloc_mode.gen.tp_size,
        pp_size=alloc_mode.gen.pp_size,
    )
    return ParallelStrategy(
        data_parallel_size=int(args_dict.get("data_parallel_size") or 1),
        tensor_parallel_size=int(args_dict.get("tensor_parallel_size") or 1),
        pipeline_parallel_size=int(args_dict.get("pipeline_parallel_size") or 1),
    )


def resolve_xccl_alloc_mode(
    config,
    alloc_mode: AllocationMode,
    *,
    train_world_size: int,
) -> AllocationMode:
    """One-stop resolution for XCCL weight-update ``AllocationMode``.

    Reads ``MONARCH_INFERENCE_XCCL_PARTICIPANTS`` env override, extracts vLLM
    runtime parallel dims, and builds the correct alloc_mode.  Logs the
    decision at INFO level.

    Parameters
    ----------
    config
        Parsed AReaL config (must have ``config.vllm``).
    alloc_mode
        Cluster-level allocation mode (inference + training topology).
    train_world_size
        FSDP / training ProcMesh world size.
    """
    flat_inf = optional_flat_inference_xccl_from_env()
    vllm_gen_ps = _vllm_gen_parallel(config, alloc_mode)
    result = build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
        vllm_gen_ps,
        gen_backend=alloc_mode.gen_backend or "vllm",
        train_world_size=train_world_size,
        flat_inference_dp=flat_inf,
    )
    if flat_inf is not None:
        logger.info(
            "XCCL weight-update: MONARCH_INFERENCE_XCCL_PARTICIPANTS=%s "
            "forces inference half to d%sp1t1 (flat mode)",
            flat_inf,
            flat_inf,
        )
    else:
        logger.info(
            "XCCL weight-update alloc_mode from embedded vLLM parallel "
            "(gen.world_size=%d; cluster inference devices=%d)",
            vllm_gen_ps.world_size,
            alloc_mode.gen.world_size,
        )
    return result
