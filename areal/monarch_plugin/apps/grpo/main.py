"""GRPO training entry point -- clean async orchestration.

Uses the proven actor implementations (GeneratorActor, TrainerActor,
RolloutActor, etc.) with TorchForge-style ``options().as_actor()``
declarative spawning and the async rollout-training pipeline.

This entry point replaces ``launcher.py`` as the canonical way to run
Monarch-based GRPO training. The actors and MonarchVLLMEngine bridge
are unchanged -- only the orchestration layer is restructured.

Usage::

    python -m areal.monarch_plugin.apps.grpo.main \
        examples/math/gsm8k_rl.py \
        --config examples/math/gsm8k_grpo_npu.yaml
"""

from __future__ import annotations

import asyncio
import os
import sys

from areal.api import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    InferenceEngineConfig,
    RecoverConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    get_scheduling_spec,
    get_thread_env_vars,
    validate_config_for_launcher,
)
from areal.monarch_plugin.bootstraps import (
    ensure_ascend_custom_opp_path,
    make_generator_bootstrap,
    make_trainer_bootstrap_multi,
)
from areal.monarch_plugin.executor import WorkerRegistry
from areal.monarch_plugin.generator_actor import GeneratorActor
from areal.monarch_plugin.pipeline import run_training_pipeline
from areal.monarch_plugin.replay_buffer_actor import ReplayBufferActor
from areal.monarch_plugin.reward_actor import RewardActor
from areal.monarch_plugin.rollout_actor import RolloutActor
from areal.monarch_plugin.trainer_actor import TrainerActor
from areal.monarch_plugin.weight_sync import resolve_xccl_alloc_mode
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports, gethostip
from areal.utils.recover import check_if_recover

logger = logging.getLogger("GRPOApp")

ensure_ascend_custom_opp_path()


async def grpo_main(config, run_id: int = 0):
    """Async GRPO training orchestration.

    Architecture::

        continuous_rollouts:  RolloutActor → ReplayBuffer
        continuous_training:  ReplayBuffer → TrainerActor
        weight_sync:          TrainerActor → GeneratorActor (XCCL)
    """
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_launcher(config)

    config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
    config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
        )
    )
    alloc_mode = AllocationMode.from_str(config.allocation_mode)

    logger.info(
        f"GRPO: experiment={config.experiment_name}, "
        f"trial={config.trial_name}, run_id={run_id}"
    )

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    fileroot = config.cluster.fileroot
    user = os.environ.get("USER", "root")
    os.makedirs(
        f"{fileroot}/logs/{user}/{config.experiment_name}/{config.trial_name}",
        exist_ok=True,
    )

    master_addr = gethostip()
    master_port = find_free_ports(1, (10000, 50000))[0]

    inf_ws = alloc_mode.gen.world_size
    train_ws = alloc_mode.train.world_size if alloc_mode.train else 0
    inf_ids_str = ",".join(str(d) for d in range(inf_ws))
    train_ids_str = ",".join(str(d) for d in range(inf_ws, inf_ws + train_ws))

    logger.info(
        f"Device partition: inference={inf_ids_str} training={train_ids_str}"
    )

    from monarch._src.actor.host_mesh import this_proc

    proc = this_proc()
    host_mesh = proc

    logger.info(">>> Spawning Worker Registry")
    worker_registry = proc.spawn("worker_registry", WorkerRegistry)

    logger.info(">>> Spawning Generator (inference)")
    generator_actor = await GeneratorActor.options(
        num_replicas=1,
        procs=1,
        with_gpus=True,
        bootstrap=make_generator_bootstrap(inf_ids_str),
    ).as_actor(GeneratorActor.build_vllm_cli_args(config, alloc_mode))

    logger.info(">>> Spawning Reward Actor")
    reward_actor = await RewardActor.options(
        num_replicas=1, procs=1, with_gpus=False
    ).as_actor()

    from monarch._src.actor.host_mesh import this_host

    await generator_actor.setup.call(this_host(), worker_registry, None)

    reward_fn_path = (
        config.get("reward_fn") or "areal.reward.gsm8k.gsm8k_reward_fn"
    )
    await reward_actor.setup.call(reward_fn_path)

    logger.info(">>> Spawning Replay Buffer + Rollout Actor")
    replay_buffer_actor = await ReplayBufferActor.options(
        procs=1, with_gpus=False
    ).as_actor()

    rollout_actor = await RolloutActor.options(
        procs=1, with_gpus=False
    ).as_actor(generator_actor, reward_actor, None)

    if alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY:
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

    logger.info(">>> Spawning Trainer (FSDP)")

    xccl_alloc = resolve_xccl_alloc_mode(
        config, alloc_mode, train_world_size=train_ws
    )

    actor_spec = get_scheduling_spec(config.actor)
    thread_env = get_thread_env_vars(
        cpus_per_task=actor_spec.cpu,
        existing_env_vars=actor_spec.env_vars,
    )
    trainer_env = {
        **BASE_ENVIRONS,
        **thread_env,
        **actor_spec.env_vars,
        "AREAL_SPMD_MODE": "1",
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
    }

    trainer_actor = await TrainerActor.options(
        num_replicas=1,
        procs=train_ws,
        with_gpus=True,
        bootstrap=make_trainer_bootstrap_multi(train_ids_str),
    ).as_actor(
        cli_args=sys.argv[1:],
        env_vars=trainer_env,
        rank=-1,
        world_size=train_ws,
        master_addr=master_addr,
        master_port=master_port,
        generator_actor=generator_actor,
        reward_actor=reward_actor,
        agent_actor=None,
        xccl_weight_update_alloc_mode=xccl_alloc,
    )

    logger.info(">>> Initializing Trainer Pipeline...")
    info_mesh = await trainer_actor.initialize.call()
    _, info = next(iter(info_mesh.items()))

    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info(f"Trainer ready: max_steps={max_steps}, start_step={start_step}")

    logger.info(">>> Setting up Rollout Actor...")
    rollout_setup_info = await rollout_actor.setup.call_one(
        sys.argv[1:], train_dp_size=train_ws
    )
    logger.info(f"RolloutActor ready: {rollout_setup_info}")

    logger.info(">>> Starting Training Pipeline...")
    await run_training_pipeline(
        rollout_actor=rollout_actor,
        replay_buffer_actor=replay_buffer_actor,
        trainer_actor=trainer_actor,
        max_steps=max_steps,
        start_step=start_step,
        multi_rank=(train_ws > 1),
    )


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    config, _ = parse_cli_args(sys.argv[1:])
    asyncio.run(grpo_main(config, run_id=0))


if __name__ == "__main__":
    main()
