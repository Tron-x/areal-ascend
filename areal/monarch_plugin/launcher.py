"""
Monarch-based launcher for AReaL -- topology-aware distributed RL.

Refactored to use purely declarative TorchForge pattern.
No more topology.py, no more ActorRegistry. Components are spawned procedurally.
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
from areal.infra.utils.launcher import validate_config_for_launcher
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports, gethostip
from areal.utils.recover import check_if_recover

logger = logging.getLogger("MonarchPlugin")

# ---------------------------------------------------------------------------
# Monarch orchestration
# ---------------------------------------------------------------------------

from areal.monarch_plugin.agent_actor import AgentActor
from areal.monarch_plugin.bootstraps import (
    ensure_ascend_custom_opp_path,
    make_generator_bootstrap,
    make_trainer_bootstrap_multi,
)
from areal.monarch_plugin.generator_actor import GeneratorActor
from areal.monarch_plugin.pipeline import run_training_pipeline
from areal.monarch_plugin.replay_buffer_actor import ReplayBufferActor
from areal.monarch_plugin.reward_actor import RewardActor
from areal.monarch_plugin.rollout_actor import RolloutActor
from areal.monarch_plugin.sandbox_actor import SandboxActor
from areal.monarch_plugin.trainer_actor import TrainerActor
from areal.monarch_plugin.executor import WorkerRegistry
from monarch._src.actor.host_mesh import this_host, this_proc

ensure_ascend_custom_opp_path()


async def monarch_main_async(config, run_id: int = 0):
    """Declarative Monarch orchestration: AReaLMonarchActor.options().as_service()"""
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_launcher(config)

    config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
    config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(experiment_name=config.experiment_name, trial_name=config.trial_name)
    )
    alloc_mode = AllocationMode.from_str(config.allocation_mode)

    logger.info(f"MonarchPlugin: experiment={config.experiment_name}, trial={config.trial_name}, run_id={run_id}")

    if not is_recover_run:
        metadata_file = save_experiment_metadata(config.cluster.fileroot, config.experiment_name, config.trial_name)
        logger.info(f"Saved experiment metadata to {metadata_file}")

    fileroot = config.cluster.fileroot
    user = os.environ.get("USER", "root")
    os.makedirs(f"{fileroot}/logs/{user}/{config.experiment_name}/{config.trial_name}", exist_ok=True)

    master_addr = gethostip()
    master_port = find_free_ports(1, (10000, 50000))[0]

    inf_ws = alloc_mode.gen.world_size
    train_ws = alloc_mode.train.world_size if alloc_mode.train else 0
    inf_device_ids = list(range(inf_ws))
    train_device_ids = list(range(inf_ws, inf_ws + train_ws))
    inf_ids_str = ",".join(str(d) for d in inf_device_ids)
    train_ids_str = ",".join(str(d) for d in train_device_ids)

    logger.info(
        f"Device partition: inference={inf_ids_str} training={train_ids_str}"
    )

    host = this_host()
    proc = this_proc()

    logger.info(">>> Spawning Worker Registry")
    worker_registry = proc.spawn("worker_registry", WorkerRegistry)

    logger.info(">>> Spawning Declarative Base Services (TorchForge Style)")

    generator_actor = await GeneratorActor.options(
        num_replicas=1,
        procs=1,
        with_gpus=True,
        bootstrap=make_generator_bootstrap(inf_ids_str),
    ).as_actor(GeneratorActor.build_vllm_cli_args(config, alloc_mode))

    reward_actor = await RewardActor.options(
        num_replicas=1, procs=1, with_gpus=False
    ).as_actor()

    # Conditional spawning of Agent and Sandbox
    # Logic: if "enable_thinking" is set in gconfig or we see an AgentWorkflow string
    enable_agent = config.get("enable_thinking", False) or "AgentWorkflow" in str(sys.argv)
    
    agent_actor = None
    sandbox_actor = None
    if enable_agent:
        logger.info(">>> Spawning Agentic Auxillaries (Sandbox & Agent)")
        sandbox_actor = await SandboxActor.options(procs=1, with_gpus=False).as_actor()
        agent_actor = await AgentActor.options(procs=1, with_gpus=False).as_actor()

    # Setup Generator
    # device_ids=None lets the actor auto-discover from Monarch environment
    await generator_actor.setup.call(host, worker_registry, None)

    # Resolve reward function
    # Default to math reward if not specified in CLI or YAML
    reward_fn_path = config.get("reward_fn") or "areal.reward.gsm8k.gsm8k_reward_fn"
    await reward_actor.setup.call(reward_fn_path)

    logger.info(">>> Spawning Auxillary Training Actors")
    replay_buffer_actor = await ReplayBufferActor.options(procs=1, with_gpus=False).as_actor()
    
    # Passing None for sandbox/agent if disabled
    rollout_actor = await RolloutActor.options(procs=1, with_gpus=False).as_actor(
        generator_actor, reward_actor, agent_actor
    )

    if alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY:
        logger.info("LLM_SERVER_ONLY mode -- skipping training pipeline")
        return

    logger.info(">>> Spawning Trainer Actor (FSDP)")
    from areal.infra.utils.launcher import BASE_ENVIRONS, get_scheduling_spec, get_thread_env_vars
    from areal.monarch_plugin.weight_sync import resolve_xccl_alloc_mode
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
        agent_actor=agent_actor,
        xccl_weight_update_alloc_mode=xccl_alloc,
    )

    logger.info(">>> Initializing Trainer Pipeline...")
    info_mesh = await trainer_actor.initialize.call()
    _, info = next(iter(info_mesh.items()))

    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info(f"Trainer ready: max_steps={max_steps}, start_step={start_step}")

    logger.info(">>> Setting up RolloutActor...")
    rollout_setup_info = await rollout_actor.setup.call_one(
        sys.argv[1:], train_dp_size=train_ws
    )
    logger.info(f"RolloutActor ready: {rollout_setup_info}")

    logger.info(">>> Emitting Training Pipeline...")
    await run_training_pipeline(
        rollout_actor=rollout_actor,
        replay_buffer_actor=replay_buffer_actor,
        trainer_actor=trainer_actor,
        max_steps=max_steps,
        start_step=start_step,
        multi_rank=(train_ws > 1),
    )


def monarch_main(config, run_id: int = 0):
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(monarch_main_async(config, run_id))


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    monarch_main(config, run_id=0)


if __name__ == "__main__":
    main()
