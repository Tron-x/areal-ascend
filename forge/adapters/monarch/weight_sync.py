"""XCCL weight synchronization for Forge on Monarch.

Wraps AReaL's XCCL (HCCL/NCCL) weight update logic as a
``WeightStore`` implementation.
"""

from __future__ import annotations

import logging
import os

from forge.core.weight_sync import WeightStore

logger = logging.getLogger("forge.monarch.weight_sync")


def optional_flat_inference_xccl_from_env() -> int | None:
    """Read ``MONARCH_INFERENCE_XCCL_PARTICIPANTS`` override."""
    raw = os.environ.get("MONARCH_INFERENCE_XCCL_PARTICIPANTS")
    if raw is None or str(raw).strip() == "":
        return None
    v = int(raw)
    if v < 1:
        raise ValueError(f"MONARCH_INFERENCE_XCCL_PARTICIPANTS must be >= 1, got {v!r}")
    return v


def resolve_xccl_alloc_mode(config, alloc_mode, *, train_world_size: int):
    """One-stop resolution for XCCL weight-update AllocationMode.

    Delegates to AReaL's allocation mode utilities.
    """
    from areal.api.alloc_mode import (
        AllocationMode,
        InferenceParallelism,
        ParallelStrategy,
    )
    from areal.api.cli_args import vLLMConfig

    flat_inf = optional_flat_inference_xccl_from_env()

    args_dict = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=alloc_mode.gen.tp_size,
        pp_size=alloc_mode.gen.pp_size,
    )
    vllm_gen_ps = ParallelStrategy(
        data_parallel_size=int(args_dict.get("data_parallel_size") or 1),
        tensor_parallel_size=int(args_dict.get("tensor_parallel_size") or 1),
        pipeline_parallel_size=int(args_dict.get("pipeline_parallel_size") or 1),
    )

    gen_backend = alloc_mode.gen_backend or "vllm"

    if flat_inf is not None:
        s = f"{gen_backend}:d{flat_inf}p1t1+d{train_world_size}p1t1"
        logger.info(
            "XCCL: MONARCH_INFERENCE_XCCL_PARTICIPANTS=%d forces flat mode", flat_inf
        )
        return AllocationMode.from_str(s)

    gen_str = str(InferenceParallelism(gen_backend, vllm_gen_ps))
    train_str = f"d{train_world_size}p1t1"
    result = AllocationMode.from_str(f"{gen_str}+{train_str}")
    logger.info(
        "XCCL alloc_mode from vLLM parallel (gen.world_size=%d)",
        vllm_gen_ps.world_size,
    )
    return result


class XCCLWeightStore(WeightStore):
    """XCCL (HCCL/NCCL)-based weight synchronization.

    Weight updates happen in-band via collective operations between
    the training and inference process groups. This store acts as
    a thin coordinator -- the actual transfer is done by FSDPEngine
    and vLLM's VLLMWorkerExtension.
    """

    def __init__(self, trainer_actor, generator_actor):
        self._trainer = trainer_actor
        self._generator = generator_actor
        self._version = 0

    async def push(self, version: int) -> None:
        self._version = version
        logger.info("XCCLWeightStore: push version %d (handled by trainer)", version)

    async def pull(self, version: int) -> None:
        logger.info("XCCLWeightStore: pull version %d (handled by generator)", version)

    async def latest_version(self) -> int:
        return self._version
