"""Agentic RL training entry point -- multi-turn agent with code execution.

Extends the GRPO pattern with AgentActor (multi-turn loops),
SandboxActor (code execution), and session-based routing for
KV cache locality.

Usage::

    python -m areal.monarch_plugin.apps.agent_rl.main \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_agent.yaml \\
        ++enable_thinking=true
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
from areal.monarch_plugin.actors.agent import AgentActor
from areal.monarch_plugin.actors.generator import Generator
from areal.monarch_plugin.actors.replay_buffer import ReplayBuffer
from areal.monarch_plugin.actors.reward import RewardActor
from areal.monarch_plugin.actors.sandbox import SandboxActor
from areal.monarch_plugin.actors.trainer import TrainerActor
from areal.monarch_plugin.bootstraps import ensure_ascend_custom_opp_path
from areal.monarch_plugin.controller.provisioner import init_provisioner, shutdown
from areal.monarch_plugin.weight_sync import resolve_xccl_alloc_mode
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports, gethostip
from areal.utils.recover import check_if_recover

logger = logging.getLogger("AgentRLApp")


async def agent_rl_main(config, run_id: int = 0):
    """Async agentic RL orchestration with multi-turn agent workflows.

    Architecture::

        continuous_rollouts:
            AgentActor.run_episode (session-affine routing)
              -> Generator.generate (text generation)
              -> SandboxActor.execute_code (code execution)
              -> RewardActor.compute_reward (scoring)
            -> ReplayBuffer.add

        continuous_training:
            ReplayBuffer.sample -> Trainer.train_on_batch
            -> Generator.update_weights
    """
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
        f"AgentRL: experiment={config.experiment_name}, "
        f"trial={config.trial_name}, run_id={run_id}"
    )

    if not is_recover_run:
        save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )

    fileroot = config.cluster.fileroot
    user = os.environ.get("USER", "root")
    os.makedirs(
        f"{fileroot}/logs/{user}/{config.experiment_name}/{config.trial_name}",
        exist_ok=True,
    )

    master_addr = gethostip()
    master_port = find_free_ports(1, (10000, 50000))[0]

    train_ws = alloc_mode.train.world_size if alloc_mode.train else 0

    await init_provisioner()

    generator = await Generator.options(
        procs=1, with_gpus=True, mesh_name="generator"
    ).as_service(
        engine_args=_build_vllm_engine_args(config, alloc_mode),
    )

    reward_fn_path = config.get("reward_fn") or "areal.reward.gsm8k.gsm8k_reward_fn"
    reward = await RewardActor.options(
        num_replicas=2, procs=1, mesh_name="reward"
    ).as_service()
    await reward.setup.fanout(reward_fn_path)

    sandbox = await SandboxActor.options(
        num_replicas=4, procs=1, mesh_name="sandbox"
    ).as_service()

    max_turns = config.get("max_turns", 3)
    turn_discount = config.get("turn_discount", 0.9)

    agent = await AgentActor.options(
        num_replicas=2, procs=1, mesh_name="agent"
    ).as_service(
        generator=generator,
        reward=reward,
        sandbox=sandbox,
        max_turns=max_turns,
        turn_discount=turn_discount,
    )

    _replay_buffer = await ReplayBuffer.options(  # noqa: F841
        procs=1, mesh_name="buffer"
    ).as_actor(max_size=4096, eviction_policy="age", max_age_steps=2)

    if alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY:
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

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

    xccl_alloc = resolve_xccl_alloc_mode(config, alloc_mode, train_world_size=train_ws)

    trainer = await TrainerActor.options(
        procs=train_ws, with_gpus=True, mesh_name="trainer"
    ).as_actor(
        cli_args=sys.argv[1:],
        env_vars=trainer_env,
        rank=-1,
        world_size=train_ws,
        master_addr=master_addr,
        master_port=master_port,
        generator_actor=generator,
        reward_actor=reward,
        agent_actor=agent,
        xccl_weight_update_alloc_mode=xccl_alloc,
    )

    info_mesh = await trainer.initialize.call()
    _, info = next(iter(info_mesh.items()))
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info(f"Trainer ready: max_steps={max_steps}, start={start_step}")

    step = start_step
    shutdown_event = asyncio.Event()

    async def continuous_training():
        nonlocal step
        while step < max_steps and not shutdown_event.is_set():
            logger.info(f"[Train] Step {step}/{max_steps}")
            result_mesh = await trainer.train_step.call(step)
            _, result = next(iter(result_mesh.items()))
            logger.info(f"[Train] Step {step} complete: {result}")
            step += 1
        shutdown_event.set()

    try:
        await continuous_training()
    except Exception as e:
        logger.error(f"Agent RL training failed: {e}")
        raise
    finally:
        logger.info("Shutting down all actors...")
        await shutdown()

    logger.info(f"Agent RL training complete. Total steps: {step}")


def _build_vllm_engine_args(config, alloc_mode) -> dict:
    vllm_args = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=alloc_mode.gen.tp_size,
        pp_size=alloc_mode.gen.pp_size,
    )
    return vllm_args


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    config, _ = parse_cli_args(sys.argv[1:])
    asyncio.run(agent_rl_main(config, run_id=0))


if __name__ == "__main__":
    main()
