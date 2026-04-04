"""Agentic RL training -- TorchForge-style async orchestration.

Two execution modes:

**Synchronous** (``async_pipeline=False``, default): rollout and training
alternate on each step via ``trainer.train_step``.

**Async pipeline** (``async_pipeline=True``): two concurrent loops
connected through a ReplayBuffer, following TorchForge's pattern:

- ``continuous_rollouts``: orchestrator calls Generator/AgentActor
  directly, computes reward, pushes ``Episode`` to ReplayBuffer
- ``continuous_training``: polls ReplayBuffer, uses ``BatchAdapter`` to
  convert Episodes to engine-specific format, trains, syncs weights

Two backend paths:

- **TrainEngine** (new): framework-agnostic. Weight sync is handled
  externally via ``WeightSyncStrategy`` (NCCL/checkpoint/HIXL).
- **TrainBackend** (legacy/AReaL): weight sync is built into the backend.

Usage::

    # Sync mode with AReaL backend (default)
    python -m forge.apps.agent_rl examples/math/gsm8k_rl.py ...

    # Async pipeline with FSDP backend
    FORGE_ASYNC_PIPELINE=1 python -m forge.apps.agent_rl examples/math/gsm8k_rl.py \\
        --backend_type fsdp --model_path Qwen/Qwen2.5-1.5B
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
from forge.core.types import Episode
from forge.engines import create_batch_adapter, create_config_bridge, create_engine
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
    data_source: list[dict] | None = None,
    use_retool: bool = False,
):
    """Produce rollout ``Episode`` objects and push them to the ReplayBuffer.

    All GPU work is offloaded to Generator / RewardActor / AgentActor via
    Monarch RPC.  Episodes are framework-agnostic -- conversion to
    engine-specific format happens in ``continuous_training`` via the
    ``BatchAdapter``.
    """
    rollout_count = 0
    data_idx = 0
    shutdown = shutdown_event or asyncio.Event()

    while not shutdown.is_set():
        try:
            data = None
            if data_source:
                data = data_source[data_idx % len(data_source)]
                data_idx += 1

            if agent is not None:
                episode = await _rollout_via_agent(
                    agent, rollout_count, data=data, use_retool=use_retool
                )
            else:
                episode = await _rollout_via_generator(generator, reward, rollout_count)

            if episode is not None:
                await replay_buffer.add.call_one(
                    episode.to_dict(), version=rollout_count, step=rollout_count
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


async def _rollout_via_agent(
    agent, step: int, data: dict | None = None, use_retool: bool = False
) -> Episode | None:
    """Run a multi-turn episode through AgentActor.

    Returns:
        An ``Episode`` object (framework-agnostic).
    """
    if data is None:
        data = {"prompt": f"step_{step}", "step": step}

    if use_retool:
        ep = agent.run_episode_retool
    else:
        ep = agent.run_episode

    result_mesh = await ep.call(data)
    _, result = next(iter(result_mesh.items()))

    if isinstance(result, Episode):
        return result
    if isinstance(result, dict):
        return Episode.from_dict(result)
    return result


async def _rollout_via_generator(generator, reward, step: int) -> Episode | None:
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

    return Episode(
        episode_id=f"gen_ep_{step}",
        prompt=prompt,
        response=text,
        reward=float(r),
        policy_version=version,
        token_ids=token_ids,
        generator_logprobs=logprobs,
        loss_mask=[1] * len(token_ids),
        versions=[version] * len(token_ids),
    )


async def continuous_training(
    trainer,
    replay_buffer,
    generator,
    max_steps: int,
    start_step: int = 0,
    shutdown_event: asyncio.Event | None = None,
    poll_interval: float = 0.5,
    batch_adapter=None,
    weight_sync=None,
    use_engine: bool = False,
):
    """Consume from ReplayBuffer, adapt via BatchAdapter, train, sync weights.

    The ``batch_adapter`` converts framework-agnostic ``Episode`` dicts
    from the ReplayBuffer into the engine-specific tensor layout.
    If no adapter is provided, raw dicts are passed through (legacy path).

    Two paths:

    - **TrainEngine** (``use_engine=True``): trains via ``train_on_engine_batch``,
      weight sync via ``weight_sync.push(version)``.
    - **TrainBackend** (legacy, ``use_engine=False``): trains via
      ``train_on_buffered_batch`` with built-in weight sync.
    """
    step = start_step
    shutdown = shutdown_event or asyncio.Event()

    while (max_steps == -1 or step < max_steps) and not shutdown.is_set():
        batch_mesh = await replay_buffer.wait_and_sample.call(
            batch_size=1, current_step=step
        )
        _, sampled = next(iter(batch_mesh.items()))

        if sampled is None:
            await asyncio.sleep(poll_interval)
            continue

        logger.info(f"[Train] Step {step}/{max_steps} — training on buffered batch")

        if batch_adapter is not None:
            episodes = [
                Episode.from_dict(d) if isinstance(d, dict) else d for d in sampled
            ]
            batch = batch_adapter.adapt(episodes)
        else:
            batch = sampled[0]

        if use_engine:
            result_mesh = await trainer.train_on_engine_batch.call(batch, step)
            _, result = next(iter(result_mesh.items()))
            if weight_sync is not None:
                await weight_sync.push(step + 1)
        else:
            result_mesh = await trainer.train_on_buffered_batch.call(
                batch, step, skip_weight_sync=False
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
# Weight sync helper
# ======================================================================


async def _create_weight_sync(forge_cfg, trainer, generator):
    """Create and initialize a WeightSyncStrategy for TrainEngine backends.

    Reads ``FORGE_WEIGHT_SYNC`` env var (default ``"nccl"``) to select
    the sync method.  Returns an initialized strategy, or None on failure.
    """
    from forge.core.weight_sync import WeightSyncConfig, WeightSyncMethod
    from forge.engines.weight_sync import create_weight_sync

    method_str = os.environ.get("FORGE_WEIGHT_SYNC", "nccl")
    method = WeightSyncMethod(method_str)

    config = WeightSyncConfig(
        method=method,
        checkpoint_dir=os.path.join(forge_cfg.fileroot, "weight_sync"),
        master_addr=forge_cfg.master_addr,
        master_port=forge_cfg.master_port + 100,
        world_size=forge_cfg.train_world_size + forge_cfg.gen_world_size,
        rank_offset=forge_cfg.train_world_size,
        backend="hccl" if os.environ.get("ASCEND_VISIBLE_DEVICES") else "nccl",
    )

    strategy = create_weight_sync(method_str)
    try:
        await strategy.initialize(trainer, generator, config)
        logger.info(f"Weight sync initialized: method={method_str}")
    except Exception as e:
        logger.warning(f"Weight sync initialization failed: {e}. Continuing without sync.")
        return None
    return strategy


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

    use_rm_gpu = forge_cfg.reward_mode in ("model", "hybrid")
    reward = await RewardActor.options(
        procs=1, with_gpus=use_rm_gpu, mesh_name="reward"
    ).as_actor()
    if forge_cfg.reward_fn_path:
        await reward.setup.call(forge_cfg.reward_fn_path)
    if forge_cfg.reward_model_path:
        await reward.setup_model.call(
            model_path=forge_cfg.reward_model_path,
            device=forge_cfg.reward_model_device,
            dtype=forge_cfg.reward_model_dtype,
            max_batch_size=forge_cfg.reward_model_max_batch_size,
            max_length=forge_cfg.reward_model_max_length,
        )
    if forge_cfg.reward_mode != "rule":
        await reward.set_mode.call(
            mode=forge_cfg.reward_mode,
            rule_weight=forge_cfg.reward_rule_weight,
            model_weight=forge_cfg.reward_model_weight,
        )

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

    use_engine = forge_cfg.backend_type != "areal"
    weight_sync = None

    if use_engine:
        engine = create_engine(backend=forge_cfg.backend_type, config={
            "model_path": forge_cfg.model_path,
            "max_steps": forge_cfg.backend_config.get("max_steps", 100),
            **forge_cfg.backend_config,
        })
        trainer = await TrainerActor.options(
            procs=forge_cfg.train_world_size, with_gpus=True, mesh_name="trainer"
        ).as_actor(engine=engine)

        weight_sync = await _create_weight_sync(forge_cfg, trainer, generator)
    else:
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
        ).as_actor(backend=backend)

    info_mesh = await trainer.initialize.call()
    _, info = next(iter(info_mesh.items()))
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info(f"Trainer ready: max_steps={max_steps}, start={start_step}")

    # -- Run -------------------------------------------------------------------

    batch_adapter = create_batch_adapter(backend=forge_cfg.backend_type)
    logger.info(f"Created batch adapter for backend={forge_cfg.backend_type!r}")

    try:
        if async_pipeline:
            shutdown_event = asyncio.Event()
            logger.info(
                f"Starting async pipeline (max_steps={max_steps}, start={start_step})"
            )

            data_source = None
            try:
                from forge.examples.retool_gsm8k.data import load_gsm8k_prompts

                data_source = load_gsm8k_prompts(
                    split="train", max_samples=256, shuffle=True
                )
                logger.info(f"Loaded {len(data_source)} real prompts for rollout")
            except Exception as e:
                logger.warning(f"Could not load GSM8K data: {e}. Using dummy prompts.")

            num_rollout_threads = forge_cfg.rollout_threads
            logger.info(
                f"Launching {num_rollout_threads} rollout thread(s) + 1 training thread"
            )

            rollout_tasks = [
                asyncio.create_task(
                    continuous_rollouts(
                        generator=generator,
                        reward=reward,
                        replay_buffer=replay_buffer,
                        agent=agent,
                        shutdown_event=shutdown_event,
                        data_source=data_source,
                        use_retool=data_source is not None,
                    )
                )
                for _ in range(num_rollout_threads)
            ]
            training_task = asyncio.create_task(
                continuous_training(
                    trainer=trainer,
                    replay_buffer=replay_buffer,
                    generator=generator,
                    max_steps=max_steps,
                    start_step=start_step,
                    shutdown_event=shutdown_event,
                    batch_adapter=batch_adapter,
                    weight_sync=weight_sync,
                    use_engine=use_engine,
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

            for i, rt in enumerate(rollout_tasks):
                rt.add_done_callback(
                    lambda t, i=i: _on_task_done(t, f"rollout_task_{i}")
                )
            training_task.add_done_callback(lambda t: _on_task_done(t, "training_task"))

            try:
                await training_task
            except Exception as e:
                logger.error(f"Training task failed: {e}")
            finally:
                shutdown_event.set()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*rollout_tasks, return_exceptions=True),
                        timeout=10,
                    )
                except TimeoutError:
                    for rt in rollout_tasks:
                        rt.cancel()
                    await asyncio.gather(*rollout_tasks, return_exceptions=True)
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
