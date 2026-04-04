"""GRPO training entry point using Forge actors.

Two backend paths:

- **TrainBackend** (legacy/AReaL): synchronous ``trainer.train_step(step)``
  with built-in rollout + weight sync.
- **TrainEngine** (new/FSDP/Megatron): uses ``BatchAdapter`` for external
  batch, ``WeightSyncStrategy`` for weight transfer.

Engine-agnostic -- the backend is selected via ``ForgeConfig.backend_type``
(default ``"areal"``), configurable to any engine in ``forge/engines/``.

Usage::

    python -m forge.apps.grpo \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml
"""

from __future__ import annotations

import asyncio
import logging
import os

from forge.actors.generator import Generator
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.reward import RewardActor
from forge.actors.trainer import TrainerActor
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.core.types import Episode
from forge.engines import create_batch_adapter, create_config_bridge, create_engine
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("ForgeGRPO")


async def _create_weight_sync(forge_cfg, trainer, generator):
    """Create and initialize a WeightSyncStrategy for TrainEngine backends."""
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
        logger.warning(f"Weight sync init failed: {e}. Continuing without sync.")
        return None
    return strategy


async def _engine_training_loop(
    trainer, generator, reward, batch_adapter, weight_sync,
    max_steps: int, start_step: int = 0,
):
    """Synchronous training loop for TrainEngine backends.

    Generates rollouts via Generator, computes rewards, adapts batch,
    trains via TrainEngine, and syncs weights via WeightSyncStrategy.
    """
    step = start_step
    while step < max_steps:
        logger.info(f"[Train-Engine] Step {step}/{max_steps}")

        prompt = f"step_{step}"
        gen_mesh = await generator.generate.call(prompt)
        _, gen_results = next(iter(gen_mesh.items()))

        gen_result = gen_results[0] if isinstance(gen_results, list) else gen_results
        text = gen_result.get("text", "")
        token_ids = gen_result.get("token_ids", [])
        logprobs = gen_result.get("logprobs", [])
        if not isinstance(logprobs, list):
            logprobs = [0.0] * len(token_ids)
        version = gen_result.get("generator_version", -1)

        r = 0.0
        try:
            reward_mesh = await reward.compute_reward.call(
                prompt=prompt, completion=text, task_data={}
            )
            _, r = next(iter(reward_mesh.items()))
        except Exception as e:
            logger.warning(f"Reward failed: {e}")

        episode = Episode(
            episode_id=f"grpo_ep_{step}",
            prompt=prompt,
            response=text,
            reward=float(r),
            policy_version=version,
            token_ids=token_ids,
            generator_logprobs=logprobs,
            loss_mask=[1] * len(token_ids),
            versions=[version] * len(token_ids),
        )

        batch = batch_adapter.adapt([episode])

        result_mesh = await trainer.train_on_engine_batch.call(batch, step)
        _, result = next(iter(result_mesh.items()))

        if weight_sync is not None:
            await weight_sync.push(step + 1)

        logger.info(f"[Train-Engine] Step {step} complete: {result}")
        step += 1


async def grpo_main(config=None, run_id: int = 0):
    """Async GRPO training orchestration with Forge actors.

    Architecture::

        Generator.as_actor()     -- inference (vLLM / SGLang / ...)
        RewardActor.as_actor()   -- reward computation
        TrainerActor.as_actor()  -- training (AReaL / Slime / TorchTitan / ...)

        Training loop:
            trainer.train_step(step) -> internal rollout + train
            -> weight sync to Generator
    """
    ensure_ascend_custom_opp_path()

    bridge = create_config_bridge(backend="areal")
    forge_cfg, raw_cfg, alloc_mode = bridge.parse_and_build(run_id=run_id)

    bridge.setup_name_resolve(raw_cfg)

    is_recover_run = forge_cfg.backend_config.get("is_recover_run", False)
    if not is_recover_run:
        bridge.save_metadata(raw_cfg)

    os.makedirs(forge_cfg.resolve_log_dir(), exist_ok=True)

    logger.info(
        f"GRPO: experiment={forge_cfg.experiment_name}, "
        f"trial={forge_cfg.trial_name}, run_id={run_id}, "
        f"backend={forge_cfg.backend_type}"
    )

    if bridge.is_llm_server_only(alloc_mode):
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

    await init_provisioner()

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
            agent_actor=None,
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

    step = start_step
    try:
        if use_engine:
            batch_adapter = create_batch_adapter(backend=forge_cfg.backend_type)
            await _engine_training_loop(
                trainer, generator, reward, batch_adapter, weight_sync,
                max_steps=max_steps, start_step=start_step,
            )
            step = max_steps
        else:
            while step < max_steps:
                logger.info(f"[Train] Step {step}/{max_steps}")
                result_mesh = await trainer.train_step.call(step)
                results = list(result_mesh.items())
                _, result = results[0]
                logger.info(f"[Train] Step {step} complete: {result}")
                step += 1
    except Exception as e:
        logger.error(f"GRPO training failed at step {step}: {e}")
        raise
    finally:
        if weight_sync is not None:
            await weight_sync.shutdown()
        logger.info("Shutting down all actors...")
        await shutdown()

    logger.info(f"GRPO training complete. Total steps: {step}")


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    asyncio.run(grpo_main(run_id=0))


if __name__ == "__main__":
    main()
