"""
Monarch-based launcher for AReaL -- topology-aware distributed RL.

Uses Monarch actors for the full training lifecycle:
  - GeneratorActor: embeds vLLM AsyncLLM with AReaLMonarchExecutor
  - RewardActor: dedicated reward computation on CPU ProcMesh
  - SandboxActor: isolated Python code execution on CPU ProcMesh
  - AgentActor: multi-turn agent orchestrating Generator+Sandbox+Reward
  - ReplayBufferActor: async experience replay for pipeline parallelism
  - RolloutActor: independent rollout production (CPU ProcMesh, no model)
  - TrainerActor: runs FSDP training in-process with step-level control
  - All communication via Monarch RPC (no HTTP)

Topology-aware placement (via topology.py):
  The launcher supports arbitrary device configurations through
  ClusterTopology.  Device placement is computed from allocation_mode
  and cluster config, not hardcoded.

XCCL weight-update alloc_mode (via weight_sync.py):
  For ``type="xccl"``, ``WeightUpdateMeta.alloc_mode.gen`` must match how many
  vLLM workers join the weight-update group (``GeneratorActor`` collective RPC).
  By default we rebuild alloc_mode from the cluster ``allocation_mode`` inference
  parallel strategy plus training world size; optional env
  ``MONARCH_INFERENCE_XCCL_PARTICIPANTS`` forces a flat ``d{N}p1t1`` inference
  half. See ``weight_sync.resolve_xccl_alloc_mode``.

  Single-node: devices linearly partitioned (inference first, then training)
  Multi-node:  MONARCH_WORKERS env var or cluster.monarch_workers config
               provides worker addresses; symmetric or role-split placement.

Supported configurations:
  allocation_mode         n_gpus_per_node    Layout
  ──────────────────────  ─────────────────  ──────────────────────────
  vllm:d1p1t1+d1p1t1     2                  1 inf + 1 train
  vllm:d2p1t1+d2p1t1     4                  2 inf + 2 train
  vllm:d1p1t1+d3p1t1     4                  1 inf + 3 train
  vllm:d4p1t1+d4p1t1     8                  4 inf + 4 train
  vllm:d2p1t1+d6p1t1     8                  2 inf + 6 train
  vllm:d1p1t4+d4p1t1     8                  1 inf (TP=4) + 4 train

Multi-node (requires MONARCH_WORKERS):
  allocation_mode         n_nodes  Layout
  ──────────────────────  ───────  ──────────────────────────
  vllm:d4p1t1+d4p1t1     2        each node: 4 inf + 4 train
  vllm:d8p1t1+d8p1t1     2        each node: 8 inf + 8 train

Usage:
    # 4+4 (8 NPU, single node):
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml

    # 1+1 (2 NPU):
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        "allocation_mode=vllm:d1p1t1+d1p1t1" \\
        "cluster.n_gpus_per_node=2"

    # 2+6 (8 NPU):
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        "allocation_mode=vllm:d2p1t1+d6p1t1"

    # Multi-node (2 nodes × 8 NPU):
    MONARCH_WORKERS=tcp://node0:29600,tcp://node1:29600 \\
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        "cluster.n_nodes=2"

Architecture (4+4 single-node example):
    MonarchOrchestrator (main process, no NPU)
      ├── GeneratorProcMesh (1 proc, NPUs 0-3 visible)
      │     ├── WorkerRegistry (actor)
      │     └── GeneratorActor → AsyncLLM → EngineCore subprocess
      │           └── AReaLMonarchExecutor → vLLM Worker ProcMesh (NPU 0-3)
      ├── RewardProcMesh (1 proc, CPU only)
      │     └── RewardActor → reward_fn() via Monarch RPC
      ├── SandboxProcMesh (1 proc, CPU only)
      │     └── SandboxActor → subprocess code execution
      ├── AgentProcMesh (1 proc, CPU only)
      │     └── AgentActor → multi-turn orchestration
      ├── ReplayBufferProcMesh (1 proc, CPU only)
      │     └── ReplayBufferActor → async experience replay
      ├── RolloutProcMesh (1 proc, CPU only)
      │     └── RolloutActor → dataloader + WorkflowExecutor + MonarchVLLMEngine
      │           ├── do_rollout() → produces batches → ReplayBuffer
      │           └── (no model loaded, inference via GeneratorActor RPC)
      └── TrainingProcMesh (nprocs processes, NPUs 4-7)
            ├── TrainerActor[rank=0] ─┐
            ├── TrainerActor[rank=1]  ├── FSDP via HCCL all-reduce
            ├── TrainerActor[rank=2]  │
            └── TrainerActor[rank=3] ─┘
                  └── train_on_batch() ← consumes batches ← ReplayBuffer
"""

from __future__ import annotations

import asyncio
import os
import sys

from areal.api import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    RecoverConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.infra.platforms import current_platform
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import validate_config_for_launcher
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports
from areal.utils.recover import check_if_recover

logger = logging.getLogger("MonarchPlugin")


# ---------------------------------------------------------------------------
# Monarch orchestration
# ---------------------------------------------------------------------------

from areal.monarch_plugin.actor_registry import ActorRegistry  # noqa: E402
from areal.monarch_plugin.actor_spec import ActorContext  # noqa: E402
from areal.monarch_plugin.bootstraps import ensure_ascend_custom_opp_path  # noqa: E402
from areal.monarch_plugin.pipeline import run_training_pipeline  # noqa: E402
from areal.monarch_plugin.specs import make_actor_specs  # noqa: E402
from areal.monarch_plugin.topology import ClusterTopology  # noqa: E402

# Initialise CANN custom OPP path early so child processes inherit it.
ensure_ascend_custom_opp_path()


async def monarch_main_async(config, run_id: int = 0):
    """Declarative Monarch orchestration: specs → registry → pipeline."""
    # --- Config & topology ---
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_launcher(config)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )
    alloc_mode = AllocationMode.from_str(config.allocation_mode)

    logger.info(
        f"MonarchPlugin: experiment={config.experiment_name}, "
        f"trial={config.trial_name}, run_id={run_id}"
    )

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    env_var = current_platform.device_control_env_var

    topology = ClusterTopology.from_config(config, alloc_mode, env_var=env_var)
    placement = topology.placement
    inf_device_ids = placement.inference.all_device_ids
    train_device_ids = placement.training.all_device_ids
    logger.info(f"Topology:\n{topology.summary()}")

    fileroot = config.cluster.fileroot
    user = os.environ.get("USER", "root")
    log_dir = f"{fileroot}/logs/{user}/{config.experiment_name}/{config.trial_name}"
    os.makedirs(log_dir, exist_ok=True)

    host = topology.create_host_mesh()
    master_port = find_free_ports(1, (10000, 50000))[0]

    # --- Build declarative actor specs ---
    specs = make_actor_specs(
        config=config,
        alloc_mode=alloc_mode,
        placement=placement,
        master_port=master_port,
        env_var=env_var,
        inf_device_ids=inf_device_ids,
        train_device_ids=train_device_ids,
    )

    # --- Build context and spawn ---
    ctx = ActorContext(
        config=config,
        alloc_mode=alloc_mode,
        placement=placement,
        host=host,
        master_port=master_port,
        extra={"host": host},
    )

    registry = ActorRegistry(specs)

    try:
        await registry.spawn_all(ctx, host)

        if alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY:
            logger.info("LLM_SERVER_ONLY mode -- skipping training pipeline")
            await registry.shutdown_all()
            return

        # --- Initialize actors ---
        info = await registry.initialize_all(ctx)
        trainer_info = info.get("trainer", {})
        max_steps = trainer_info["max_steps"]
        start_step = trainer_info["start_step"]

        # --- Run async training pipeline ---
        trainer_actor = registry.actor("trainer")
        rollout_actor = registry.actor("rollout")
        replay_buffer_actor = registry.actor("replay_buffer")
        multi_rank = any(s.name == "trainer" and s.is_multi_rank for s in specs)

        await run_training_pipeline(
            rollout_actor=rollout_actor,
            replay_buffer_actor=replay_buffer_actor,
            trainer_actor=trainer_actor,
            max_steps=max_steps,
            start_step=start_step,
            multi_rank=multi_rank,
        )

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down ...")
    except Exception as e:
        logger.error(f"MonarchPlugin error: {e}", exc_info=True)
        raise
    finally:
        await registry.shutdown_all()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def monarch_main(config, run_id: int = 0):
    from monarch._src.actor.actor_mesh import context

    context()
    asyncio.run(monarch_main_async(config, run_id))


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    monarch_main(config, run_id=0)


if __name__ == "__main__":
    main()
