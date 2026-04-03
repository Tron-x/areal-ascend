"""GRPO training app for Forge.

Provides a transparent, hackable training loop inspired by Slime's
``train.py`` -- the user can see and modify every step.

Two usage patterns:

1. Standalone entry point with Monarch backend::

    python -m forge.apps.grpo \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml

2. Called from ``ForgeApp.run()`` with a registered ``@rollout_fn``.

Architecture::

    continuous_rollouts:  RolloutActor --> ReplayBuffer
    continuous_training:  ReplayBuffer --> TrainerActor
    weight_sync:          TrainerActor --> GeneratorActor (XCCL)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass

logger = logging.getLogger("forge.apps.grpo")


@dataclass
class PipelineConfig:
    """Tunable parameters for the async training pipeline."""

    max_staleness: int = 3
    replay_buffer_max_size: int = 8


async def run_pipeline(
    rollout_actor,
    replay_buffer_actor,
    trainer_actor,
    *,
    max_steps: int,
    start_step: int = 0,
    multi_rank: bool = False,
    config: PipelineConfig | None = None,
) -> None:
    """Run the rollout + training pipeline until all steps complete.

    This is the core training loop -- transparent and hackable.
    Users can copy this function and modify it for custom pipelines.
    """
    if config is None:
        config = PipelineConfig()

    rollout_step = start_step
    train_step = start_step
    rollout_done = asyncio.Event()

    async def _rollout_loop():
        nonlocal rollout_step
        while rollout_step < max_steps:
            step = rollout_step
            logger.info("[Rollout] Step %d: producing batch", step)
            batch_data = await rollout_actor.do_rollout.call_one(step)
            await replay_buffer_actor.add_batch.call_one(batch_data, step)
            buf_size = await replay_buffer_actor.buffer_size.call_one()
            logger.info("[Rollout] Step %d: buffer_size=%d", step, buf_size)
            rollout_step += 1
        rollout_done.set()

    async def _training_loop():
        nonlocal train_step
        while train_step < max_steps:
            batch = await replay_buffer_actor.sample_batch.call_one(
                train_step, config.max_staleness
            )
            if batch is None:
                if rollout_done.is_set():
                    logger.warning("[Training] Buffer empty and rollout done")
                    break
                await asyncio.sleep(0.5)
                continue

            step = train_step
            if multi_rank:
                result_mesh = await trainer_actor.train_on_batch.call(batch, step)
                result = result_mesh.item(npu=0)
            else:
                result = await trainer_actor.train_on_batch.call_one(batch, step)

            logger.info(
                "[Step %d/%d] epoch=%d, epoch_step=%d",
                step + 1,
                max_steps,
                result["epoch"],
                result["epoch_step"],
            )
            train_step += 1

    rollout_task = asyncio.create_task(_rollout_loop(), name="rollout")
    train_task = asyncio.create_task(_training_loop(), name="training")

    done, pending = await asyncio.wait(
        [rollout_task, train_task], return_when=asyncio.FIRST_EXCEPTION
    )

    for task in done:
        if task.exception() is not None:
            logger.error("Task '%s' failed: %s", task.get_name(), task.exception())
            for p in pending:
                p.cancel()
            raise task.exception()

    if not (rollout_task.done() and train_task.done()):
        await asyncio.gather(rollout_task, train_task)

    buf_stats = await replay_buffer_actor.get_stats.call_one()
    logger.info("ReplayBuffer final stats: %s", buf_stats)
    logger.info("Training completed successfully.")


async def grpo_main(config, run_id: int = 0):
    """Full GRPO training orchestration on Monarch.

    This is the Monarch-specific entry point that creates all actors
    and runs the pipeline. It uses the proven actor implementations
    from ``areal.monarch_plugin``.
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

    # -- Spawn actors using Forge's MonarchForgeActor pattern --

    from monarch._src.actor.host_mesh import this_host, this_proc

    proc = this_proc()

    from forge.adapters.monarch.executor import WorkerRegistry

    worker_registry = proc.spawn("worker_registry", WorkerRegistry)

    from areal.monarch_plugin.generator_actor import GeneratorActor
    from areal.monarch_plugin.replay_buffer_actor import ReplayBufferActor
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

    logger.info(">>> Spawning ReplayBuffer + RolloutActor")
    replay_buffer_actor = await ReplayBufferActor.options(
        procs=1, with_gpus=False
    ).as_actor()
    rollout_actor = await RolloutActor.options(procs=1, with_gpus=False).as_actor(
        generator_actor, reward_actor, None
    )

    if alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY:
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

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

    logger.info(">>> Initializing Trainer")
    info_mesh = await trainer_actor.initialize.call()
    _, info = next(iter(info_mesh.items()))
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info("Trainer ready: max_steps=%d, start_step=%d", max_steps, start_step)

    logger.info(">>> Setting up RolloutActor")
    rollout_setup_info = await rollout_actor.setup.call_one(
        sys.argv[1:], train_dp_size=train_ws
    )
    logger.info("RolloutActor ready: %s", rollout_setup_info)

    logger.info(">>> Starting Training Pipeline")
    await run_pipeline(
        rollout_actor=rollout_actor,
        replay_buffer_actor=replay_buffer_actor,
        trainer_actor=trainer_actor,
        max_steps=max_steps,
        start_step=start_step,
        multi_rank=(train_ws > 1),
    )


def main():
    """CLI entry point for GRPO training."""
    from monarch._src.actor.actor_mesh import context

    from areal.api.cli_args import parse_cli_args

    context()

    config, _ = parse_cli_args(sys.argv[1:])
    asyncio.run(grpo_main(config, run_id=0))


if __name__ == "__main__":
    main()
