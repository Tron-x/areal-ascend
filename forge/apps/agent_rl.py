"""Agentic RL training with async rollout/train pipeline.

Supports two execution modes:

**Synchronous** (``async_pipeline=False``, default): rollout and training
alternate on each step, same as the original GRPO loop.

**Async pipeline** (``async_pipeline=True``): rollout and training run as
independent concurrent loops connected through a ReplayBuffer, achieving
near-2x GPU utilization by keeping Generator and Trainer GPUs busy
simultaneously.

Architecture (async mode)::

    rollout_producer_loop:
        DataProvider.get_batch -> RolloutProducer.produce_step
          -> AgentActor.run_episode (Generator GPUs)
          -> ReplayBuffer.add_batch

    train_consumer_loop:
        ReplayBuffer.wait_and_sample -> TrainerActor.train_on_buffered_batch
          (Trainer GPUs) -> TrainerActor.sync_weights -> Generator

Usage::

    python -m forge.apps.agent_rl \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_agent.yaml
"""

from __future__ import annotations

import asyncio
import logging
import os

from forge.actors.agent import AgentActor
from forge.actors.generator import Generator
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.reward import RewardActor
from forge.actors.rollout_producer import RolloutProducer
from forge.actors.sandbox import SandboxActor
from forge.actors.trainer import TrainerActor
from forge.adapters.areal import AReaLConfigBridge, AReaLTrainBackend
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("AgentRLApp")


# ======================================================================
# Async pipeline loops
# ======================================================================


async def rollout_producer_loop(
    data_provider,
    rollout_producer,
    max_steps: int,
    shutdown_event: asyncio.Event,
):
    """Continuously produce rollout batches and push to ReplayBuffer.

    Runs on Generator GPUs (via AgentActor/Generator RPC).
    """
    step = 0
    while step < max_steps and not shutdown_event.is_set():
        try:
            data_mesh = await data_provider.get_batch.call()
            data_items = next(iter(data_mesh.values()))
            if not data_items:
                logger.warning("[Rollout] DataProvider returned empty batch, retrying")
                await asyncio.sleep(1.0)
                continue

            result_mesh = await rollout_producer.produce_step.call(
                data_items, step=step, version=step
            )
            _, result = next(iter(result_mesh.items()))
            logger.info(
                f"[Rollout] Step {step}: produced {result.get('produced', 0)} "
                f"episodes in {result.get('elapsed', 0):.1f}s"
            )
            step += 1
        except Exception as e:
            logger.error(f"[Rollout] Error at step {step}: {e}")
            if shutdown_event.is_set():
                break
            await asyncio.sleep(2.0)

    logger.info(f"[Rollout] Producer finished after {step} steps")


async def train_consumer_loop(
    trainer,
    replay_buffer,
    max_steps: int,
    start_step: int,
    shutdown_event: asyncio.Event,
    poll_interval: float = 1.0,
):
    """Continuously sample from ReplayBuffer and train.

    Runs on Trainer GPUs.
    """
    step = start_step
    while step < max_steps and not shutdown_event.is_set():
        batch_mesh = await replay_buffer.wait_and_sample.call(
            batch_size=1, current_step=step
        )
        _, batch = next(iter(batch_mesh.items()))

        if batch is None:
            await asyncio.sleep(poll_interval)
            continue

        logger.info(f"[Train] Step {step}/{max_steps} — got batch from buffer")

        result_mesh = await trainer.train_on_buffered_batch.call(
            batch[0], step, skip_weight_sync=False
        )
        results = list(result_mesh.items())
        _, result = results[0]
        logger.info(f"[Train] Step {step} complete: {result}")
        step += 1

    shutdown_event.set()
    logger.info(f"[Train] Consumer finished after {step - start_step} steps")


# ======================================================================
# Synchronous fallback loop
# ======================================================================


async def sync_training_loop(
    trainer,
    max_steps: int,
    start_step: int,
):
    """Original synchronous rollout+train loop (no pipeline)."""
    step = start_step
    while step < max_steps:
        logger.info(f"[Train] Step {step}/{max_steps}")
        result_mesh = await trainer.train_step.call(step)
        results = list(result_mesh.items())
        _, result = results[0]
        logger.info(f"[Train] Step {step} complete: {result}")
        step += 1
    return step


# ======================================================================
# Main entry point
# ======================================================================


async def agent_rl_main(config=None, run_id: int = 0):
    """Agentic RL orchestration with optional async pipeline."""
    ensure_ascend_custom_opp_path()

    bridge = AReaLConfigBridge()
    forge_cfg, raw_cfg, alloc_mode = bridge.parse_and_build(run_id=run_id)

    bridge.setup_name_resolve(raw_cfg)

    is_recover_run = forge_cfg.backend_config.get("is_recover_run", False)
    if not is_recover_run:
        bridge.save_metadata(raw_cfg)

    os.makedirs(forge_cfg.resolve_log_dir(), exist_ok=True)

    async_pipeline = (
        forge_cfg.async_pipeline or os.environ.get("FORGE_ASYNC_PIPELINE", "") == "1"
    )
    logger.info(
        f"AgentRL: experiment={forge_cfg.experiment_name}, "
        f"trial={forge_cfg.trial_name}, run_id={run_id}, "
        f"async_pipeline={async_pipeline}"
    )

    await init_provisioner()

    # -- Infrastructure actors -------------------------------------------------
    # Deploy as plain actors (ActorMesh) rather than services (ServiceInterface)
    # because Monarch's ActorMesh is picklable across process boundaries,
    # while ServiceInterface contains asyncio.Future objects that cannot
    # be serialized.

    generator = await Generator.options(
        procs=1, with_gpus=True, mesh_name="generator"
    ).as_actor(
        engine_args=forge_cfg.engine_args,
    )

    reward = await RewardActor.options(procs=1, mesh_name="reward").as_actor()
    await reward.setup.call(forge_cfg.reward_fn_path)

    sandbox = await SandboxActor.options(procs=1, mesh_name="sandbox").as_actor()

    # -- Agent Framework layer -------------------------------------------------
    # NOTE: AgentActor is deployed as a single actor (not a multi-replica
    # service) because Monarch ServiceInterface objects contain asyncio
    # Futures that cannot be pickled across process boundaries. The
    # RolloutProducer handles parallelism instead.

    max_turns = raw_cfg.get("max_turns", 3) if hasattr(raw_cfg, "get") else 3
    turn_discount = (
        raw_cfg.get("turn_discount", 0.9) if hasattr(raw_cfg, "get") else 0.9
    )

    agent = await AgentActor.options(procs=1, mesh_name="agent").as_actor(
        generator=generator,
        reward=reward,
        sandbox=sandbox,
        max_turns=max_turns,
        turn_discount=turn_discount,
    )

    # -- ReplayBuffer ----------------------------------------------------------

    replay_buffer = await ReplayBuffer.options(procs=1, mesh_name="buffer").as_actor(
        max_size=forge_cfg.replay_buffer_size,
        eviction_policy="age",
        max_age_steps=forge_cfg.max_staleness_steps,
    )

    # -- Check server-only mode ------------------------------------------------

    if bridge.is_llm_server_only(alloc_mode):
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

    # -- Training backend ------------------------------------------------------

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

    # -- Run training ----------------------------------------------------------

    try:
        if async_pipeline:
            await _run_async_pipeline(
                forge_cfg=forge_cfg,
                generator=generator,
                reward=reward,
                agent=agent,
                replay_buffer=replay_buffer,
                trainer=trainer,
                max_steps=max_steps,
                start_step=start_step,
            )
        else:
            await sync_training_loop(trainer, max_steps, start_step)
    except Exception as e:
        logger.error(f"Agent RL training failed: {e}")
        raise
    finally:
        logger.info("Shutting down all actors...")
        await shutdown()

    logger.info("Agent RL training complete.")


async def _run_async_pipeline(
    *,
    forge_cfg,
    generator,
    reward,
    agent,
    replay_buffer,
    trainer,
    max_steps: int,
    start_step: int,
):
    """Set up and run the async rollout/train pipeline."""
    from forge.adapters.areal.data_provider import AReaLDataProvider

    data_provider = await AReaLDataProvider.options(
        procs=1, mesh_name="data_provider"
    ).as_actor(
        cli_args=forge_cfg.training_args,
        env_vars=forge_cfg.trainer_env,
    )
    await data_provider.setup.call()

    n_samples = 1
    rollout_producer = await RolloutProducer.options(
        procs=1, mesh_name="rollout_producer"
    ).as_actor(
        generator=generator,
        reward=reward,
        agent=agent,
        replay_buffer=replay_buffer,
        n_samples=n_samples,
    )

    shutdown_event = asyncio.Event()

    logger.info(
        f"Starting async pipeline: rollout producer + train consumer "
        f"(max_steps={max_steps}, start={start_step})"
    )

    await asyncio.gather(
        rollout_producer_loop(
            data_provider=data_provider,
            rollout_producer=rollout_producer,
            max_steps=max_steps,
            shutdown_event=shutdown_event,
        ),
        train_consumer_loop(
            trainer=trainer,
            replay_buffer=replay_buffer,
            max_steps=max_steps,
            start_step=start_step,
            shutdown_event=shutdown_event,
        ),
    )


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    asyncio.run(agent_rl_main(run_id=0))


if __name__ == "__main__":
    main()
