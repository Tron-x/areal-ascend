"""Monarch-specific actor creation and orchestration for Forge.

This is the ONLY place in the codebase that imports Monarch APIs and
AReaL legacy actors. Everything above this layer (pipeline, ForgeApp)
only sees Protocol objects.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from forge.adapters.monarch.stages import MonarchRolloutStage, MonarchTrainStage
from forge.core.replay_buffer import ReplayBuffer

logger = logging.getLogger("forge.monarch.setup")


async def setup_grpo(
    config: Any,
    run_id: int = 0,
) -> dict[str, Any]:
    """Create all Monarch actors for GRPO and wrap them as Protocols.

    Returns a dict with keys:
    - ``rollout``: ``RolloutStage``
    - ``train``: ``TrainStage``
    - ``buffer``: ``ReplayBuffer`` (local, in-process)
    - ``max_steps``: int
    - ``start_step``: int
    """
    from forge.adapters.monarch.provisioner import (
        ensure_ascend_custom_opp_path,
        make_generator_bootstrap,
        make_trainer_bootstrap_multi,
    )
    from forge.adapters.monarch.weight_sync import resolve_xccl_alloc_mode

    from areal.api import AllocationMode, AllocationType
    from areal.api.cli_args import (
        ClusterSpecConfig,
        InferenceEngineConfig,
        RecoverConfig,
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
    from areal.utils import name_resolve, names
    from areal.utils.network import find_free_ports, gethostip
    from areal.utils.recover import check_if_recover

    ensure_ascend_custom_opp_path()

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
        "GRPO: experiment=%s, trial=%s, run_id=%d",
        config.experiment_name,
        config.trial_name,
        run_id,
    )

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info("Saved experiment metadata to %s", metadata_file)

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
        "Device partition: inference=%s training=%s", inf_ids_str, train_ids_str
    )

    # -- Spawn Monarch actors (all Monarch-specific code lives here) --

    from monarch._src.actor.host_mesh import this_proc

    proc = this_proc()

    from forge.adapters.monarch.executor import WorkerRegistry

    worker_registry = proc.spawn("worker_registry", WorkerRegistry)

    from monarch._src.actor.host_mesh import this_host

    from areal.monarch_plugin.generator_actor import GeneratorActor
    from areal.monarch_plugin.reward_actor import RewardActor
    from areal.monarch_plugin.rollout_actor import RolloutActor
    from areal.monarch_plugin.trainer_actor import TrainerActor

    logger.info(">>> Spawning Generator")
    generator_actor = await GeneratorActor.options(
        num_replicas=1,
        procs=1,
        with_gpus=True,
        bootstrap=make_generator_bootstrap(inf_ids_str),
    ).as_actor(GeneratorActor.build_vllm_cli_args(config, alloc_mode))

    logger.info(">>> Spawning Reward Actor")
    reward_actor = await RewardActor.options(
        num_replicas=1,
        procs=1,
        with_gpus=False,
    ).as_actor()

    await generator_actor.setup.call(this_host(), worker_registry, None)

    reward_fn_path = config.get("reward_fn") or "areal.reward.gsm8k.gsm8k_reward_fn"
    await reward_actor.setup.call(reward_fn_path)

    logger.info(">>> Spawning RolloutActor")
    rollout_actor = await RolloutActor.options(procs=1, with_gpus=False).as_actor(
        generator_actor, reward_actor, None
    )

    if alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY:
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        raise SystemExit(0)

    logger.info(">>> Spawning Trainer (FSDP)")

    xccl_alloc = resolve_xccl_alloc_mode(config, alloc_mode, train_world_size=train_ws)

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

    # -- Wrap as Protocol objects --

    rollout = MonarchRolloutStage(rollout_actor)
    train = MonarchTrainStage(trainer_actor, multi_rank=(train_ws > 1))
    buffer = ReplayBuffer(max_size=8)

    # -- Initialize trainer and get pipeline metadata --

    logger.info(">>> Initializing Trainer")
    info = await train.get_info()
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info("Trainer ready: max_steps=%d, start_step=%d", max_steps, start_step)

    logger.info(">>> Setting up RolloutActor")
    rollout_setup_info = await rollout_actor.setup.call_one(
        sys.argv[1:], train_dp_size=train_ws
    )
    logger.info("RolloutActor ready: %s", rollout_setup_info)

    return {
        "rollout": rollout,
        "train": train,
        "buffer": buffer,
        "max_steps": max_steps,
        "start_step": start_step,
    }
