"""Agentic RL training -- TorchForge-style async orchestration.

Two execution modes:

**Synchronous** (``async_pipeline=False``, default): rollout and training
alternate on each step via ``trainer.train_step``.

**Async pipeline** (``async_pipeline=True``): two concurrent loops
connected through a ReplayBuffer, following TorchForge's pattern:

- ``continuous_rollouts``: orchestrator calls Generator/AgentActor
  directly, computes reward, pushes Episode to ReplayBuffer
- ``continuous_training``: polls ReplayBuffer, trains, syncs weights

No intermediate DataProvider or RolloutProducer actors needed -- the
orchestrator drives everything, while Generator/Reward/Sandbox actors
handle the heavy GPU work.

Usage::

    # Sync mode (default)
    python -m forge.apps.agent_rl examples/math/gsm8k_rl.py ...

    # Async pipeline mode
    FORGE_ASYNC_PIPELINE=1 python -m forge.apps.agent_rl examples/math/gsm8k_rl.py ...
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback

from forge.actors.agent import AgentActor
from forge.actors.generator import Generator
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.reward import RewardActor
from forge.actors.sandbox import SandboxActor
from forge.actors.trainer import TrainerActor
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.engines import create_config_bridge, create_engine
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("AgentRLApp")


# ======================================================================
# Async pipeline: TorchForge-style concurrent loops
# ======================================================================


async def continuous_rollouts(
    generator,
    reward,
    replay_buffer,
    agent=None,
    shutdown_event: asyncio.Event | None = None,
    n_samples: int = 1,
):
    """Produce rollout episodes and push them to the ReplayBuffer.

    Runs in the orchestrator process. All GPU work is offloaded to
    Generator / RewardActor / AgentActor via Monarch RPC.

    Follows TorchForge's ``continuous_rollouts`` pattern:
    ``generator.generate`` -> reward -> ``replay_buffer.add``.
    """
    rollout_count = 0
    shutdown = shutdown_event or asyncio.Event()

    while not shutdown.is_set():
        try:
            if agent is not None:
                episode = await _rollout_via_agent(agent, rollout_count)
            else:
                episode = await _rollout_via_generator(generator, reward, rollout_count)

            if episode is not None:
                episode = _ensure_training_keys(episode)
                await replay_buffer.add.call_one(
                    episode, version=rollout_count, step=rollout_count
                )
                rollout_count += 1
                if rollout_count % 10 == 0:
                    logger.info(f"[Rollout] Produced {rollout_count} episodes")

        except Exception as e:
            logger.error(f"[Rollout] Error at episode {rollout_count}: {e}")
            traceback.print_exc()
            if shutdown.is_set():
                break
            await asyncio.sleep(1.0)

    logger.info(f"[Rollout] Finished after {rollout_count} episodes")


def _ensure_training_keys(episode: dict) -> dict:
    """Ensure the episode dict has all keys PPOTrainer expects."""
    ids = episode.get("input_ids", [])
    seq_len = len(ids) if isinstance(ids, list) else 0

    if "attention_mask" not in episode:
        episode["attention_mask"] = [1] * seq_len
    if "loss_mask" not in episode:
        episode["loss_mask"] = [1] * seq_len
    if "logprobs" not in episode:
        episode["logprobs"] = [0.0] * seq_len
    if "versions" not in episode:
        episode["versions"] = [-1] * seq_len

    for key in ("input_ids", "logprobs", "loss_mask", "versions", "attention_mask"):
        val = episode.get(key, [])
        if isinstance(val, list) and (not val or not isinstance(val[0], list)):
            episode[key] = [val]

    reward = episode.get("rewards", 0.0)
    if not isinstance(reward, list):
        episode["rewards"] = [float(reward)]

    return episode


async def _rollout_via_agent(agent, step: int) -> dict | None:
    """Run a multi-turn episode through AgentActor."""
    data = {"prompt": f"step_{step}", "step": step}
    result_mesh = await agent.run_episode.call(data)
    _, result = next(iter(result_mesh.items()))
    return result


async def _rollout_via_generator(generator, reward, step: int) -> dict | None:
    """Single-turn generation + reward (no AgentActor)."""
    prompt = f"step_{step}"
    gen_mesh = await generator.generate.call(prompt)
    _, gen_results = next(iter(gen_mesh.items()))

    if isinstance(gen_results, list) and gen_results:
        gen_result = gen_results[0]
    elif isinstance(gen_results, dict):
        gen_result = gen_results
    else:
        return None

    text = gen_result.get("text", "")
    token_ids = gen_result.get("token_ids", [])
    logprobs = gen_result.get("logprobs", [])
    version = gen_result.get("generator_version", -1)

    if not isinstance(logprobs, list):
        logprobs = [0.0] * len(token_ids)

    r = 0.0
    try:
        reward_mesh = await reward.compute_reward.call(
            prompt=prompt, completion=text, task_data={}
        )
        _, r = next(iter(reward_mesh.items()))
    except Exception as e:
        logger.warning(f"[Rollout] Reward computation failed: {e}, returning 0.0")

    return {
        "input_ids": token_ids,
        "logprobs": logprobs,
        "loss_mask": [1] * len(token_ids),
        "versions": [version] * len(token_ids),
        "attention_mask": [1] * len(token_ids),
        "rewards": float(r),
    }


async def continuous_training(
    trainer,
    replay_buffer,
    generator,
    max_steps: int,
    start_step: int = 0,
    shutdown_event: asyncio.Event | None = None,
    poll_interval: float = 0.5,
):
    """Consume from ReplayBuffer, train, sync weights.

    Follows TorchForge's ``continuous_training`` pattern:
    ``replay_buffer.sample`` -> ``trainer.train_step`` ->
    ``trainer.push_weights`` -> ``generator.update_weights``.
    """
    step = start_step
    shutdown = shutdown_event or asyncio.Event()

    while (max_steps == -1 or step < max_steps) and not shutdown.is_set():
        batch_mesh = await replay_buffer.wait_and_sample.call(
            batch_size=1, current_step=step
        )
        _, batch = next(iter(batch_mesh.items()))

        if batch is None:
            await asyncio.sleep(poll_interval)
            continue

        logger.info(f"[Train] Step {step}/{max_steps} — training on buffered batch")

        result_mesh = await trainer.train_on_buffered_batch.call(
            batch[0], step, skip_weight_sync=False
        )
        _, result = next(iter(result_mesh.items()))

        logger.info(f"[Train] Step {step} complete: {result}")
        step += 1

    shutdown.set()
    logger.info(f"[Train] Finished after {step - start_step} steps")


# ======================================================================
# Synchronous fallback
# ======================================================================


async def sync_training_loop(trainer, max_steps: int, start_step: int):
    """Original synchronous rollout+train loop."""
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
# Main
# ======================================================================


async def agent_rl_main(config=None, run_id: int = 0):
    """Agentic RL orchestration with optional async pipeline."""
    ensure_ascend_custom_opp_path()

    bridge = create_config_bridge(backend="areal")
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

    # -- Actors ----------------------------------------------------------------

    generator = await Generator.options(
        procs=1, with_gpus=True, mesh_name="generator"
    ).as_actor(
        engine_args=forge_cfg.engine_args,
    )

    reward = await RewardActor.options(procs=1, mesh_name="reward").as_actor()
    await reward.setup.call(forge_cfg.reward_fn_path)

    sandbox = await SandboxActor.options(procs=1, mesh_name="sandbox").as_actor()

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

    replay_buffer = await ReplayBuffer.options(procs=1, mesh_name="buffer").as_actor(
        max_size=forge_cfg.replay_buffer_size,
        eviction_policy="age",
        max_age_steps=forge_cfg.max_staleness_steps,
    )

    # -- Training backend ------------------------------------------------------

    if bridge.is_llm_server_only(alloc_mode):
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

    xccl_alloc = bridge.resolve_xccl_alloc_mode(
        raw_cfg, alloc_mode, train_world_size=forge_cfg.train_world_size
    )

    backend = create_engine(
        backend=forge_cfg.backend_type,
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

    # -- Run -------------------------------------------------------------------

    try:
        if async_pipeline:
            shutdown_event = asyncio.Event()
            logger.info(
                f"Starting async pipeline (max_steps={max_steps}, start={start_step})"
            )

            rollout_task = asyncio.create_task(
                continuous_rollouts(
                    generator=generator,
                    reward=reward,
                    replay_buffer=replay_buffer,
                    agent=agent,
                    shutdown_event=shutdown_event,
                )
            )
            training_task = asyncio.create_task(
                continuous_training(
                    trainer=trainer,
                    replay_buffer=replay_buffer,
                    generator=generator,
                    max_steps=max_steps,
                    start_step=start_step,
                    shutdown_event=shutdown_event,
                )
            )

            def _on_task_done(task: asyncio.Task, name: str):
                if task.cancelled():
                    return
                exc = task.exception()
                if exc:
                    logger.error(f"{name} failed: {type(exc).__name__}: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    shutdown_event.set()

            rollout_task.add_done_callback(lambda t: _on_task_done(t, "rollout_task"))
            training_task.add_done_callback(lambda t: _on_task_done(t, "training_task"))

            try:
                await training_task
            except Exception as e:
                logger.error(f"Training task failed: {e}")
            finally:
                shutdown_event.set()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(rollout_task, return_exceptions=True),
                        timeout=10,
                    )
                except TimeoutError:
                    rollout_task.cancel()
                    await asyncio.gather(rollout_task, return_exceptions=True)
        else:
            await sync_training_loop(trainer, max_steps, start_step)
    except Exception as e:
        logger.error(f"Agent RL training failed: {e}")
        raise
    finally:
        logger.info("Shutting down all actors...")
        await shutdown()

    logger.info("Agent RL training complete.")


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    asyncio.run(agent_rl_main(run_id=0))


if __name__ == "__main__":
    main()
