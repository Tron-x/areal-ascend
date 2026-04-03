"""Agentic RL training entry point -- multi-turn agent with code execution.

Extends the GRPO pattern with AgentActor (multi-turn loops),
SandboxActor (code execution), and session-based routing for
KV cache locality.

Usage::

    python -m forge.apps.agent_rl \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_agent.yaml \\
        ++enable_thinking=true
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from forge.actors.agent import AgentActor
from forge.actors.generator import Generator
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.reward import RewardActor
from forge.actors.sandbox import SandboxActor
from forge.actors.trainer import TrainerActor
from forge.adapters.areal import AReaLConfigBridge, AReaLTrainBackend
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("AgentRLApp")


async def agent_rl_main(config=None, run_id: int = 0):
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

    bridge = AReaLConfigBridge()
    forge_cfg, raw_cfg, alloc_mode = bridge.parse_and_build(run_id=run_id)

    bridge.setup_name_resolve(raw_cfg)

    is_recover_run = forge_cfg.backend_config.get("is_recover_run", False)
    if not is_recover_run:
        bridge.save_metadata(raw_cfg)

    os.makedirs(forge_cfg.resolve_log_dir(), exist_ok=True)

    logger.info(
        f"AgentRL: experiment={forge_cfg.experiment_name}, "
        f"trial={forge_cfg.trial_name}, run_id={run_id}"
    )

    await init_provisioner()

    generator = await Generator.options(
        procs=1, with_gpus=True, mesh_name="generator"
    ).as_service(
        engine_args=forge_cfg.engine_args,
    )

    reward = await RewardActor.options(
        num_replicas=2, procs=1, mesh_name="reward"
    ).as_service()
    await reward.setup.fanout(forge_cfg.reward_fn_path)

    sandbox = await SandboxActor.options(
        num_replicas=4, procs=1, mesh_name="sandbox"
    ).as_service()

    max_turns = raw_cfg.get("max_turns", 3) if hasattr(raw_cfg, "get") else 3
    turn_discount = raw_cfg.get("turn_discount", 0.9) if hasattr(raw_cfg, "get") else 0.9

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

    if bridge.is_llm_server_only(alloc_mode):
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

    xccl_alloc = bridge.resolve_xccl_alloc_mode(
        raw_cfg, alloc_mode, train_world_size=forge_cfg.train_world_size
    )

    backend = AReaLTrainBackend(
        cli_args=forge_cfg.training_args,
        env_vars=forge_cfg.trainer_env,
        rank=-1,
        world_size=forge_cfg.train_world_size,
        master_addr=forge_cfg.master_addr,
        master_port=forge_cfg.master_port,
        generator_actor=generator,
        reward_actor=reward,
        agent_actor=agent,
        xccl_weight_update_alloc_mode=xccl_alloc,
    )

    trainer = await TrainerActor.options(
        procs=forge_cfg.train_world_size, with_gpus=True, mesh_name="trainer"
    ).as_actor(
        backend=backend,
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
            results = list(result_mesh.items())
            _, result = results[0]
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


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    asyncio.run(agent_rl_main(run_id=0))


if __name__ == "__main__":
    main()
