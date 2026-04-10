"""Standalone FSDP GRPO training -- zero AReaL dependency.

A self-contained GRPO training entry point using only:
- PyTorch FSDP2 for distributed training
- vLLM for inference
- Forge's rl/ for GRPO loss
- Forge's actors for orchestration
- Monarch for process management

This is the "AReaL-free" path that proves Forge can train independently.

Usage::

    python -m forge.apps.grpo_fsdp \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --total-train-steps 2

    # Or with the FSDP config bridge via the generic grpo app:
    python -m forge.apps.grpo --backend fsdp ...
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from forge.actors.generator import Generator
from forge.actors.reward import RewardActor
from forge.actors.trainer import TrainerActor
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.core.types import Episode
from forge.engines import create_batch_adapter, create_config_bridge, create_engine
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("GRPO-FSDP")


async def _create_weight_sync(forge_cfg, trainer, generator):
    """Create weight sync strategy for FSDP backend."""
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
        logger.info("Weight sync initialized: method=%s", method_str)
    except Exception as e:
        logger.warning("Weight sync init failed: %s", e)
        return None
    return strategy


async def _engine_training_loop(
    trainer, generator, reward, batch_adapter, weight_sync,
    max_steps: int, start_step: int = 0,
):
    """Training loop using TrainEngine (FSDP) backend."""
    step = start_step
    while step < max_steps:
        logger.info("[FSDP-Train] Step %d/%d", step, max_steps)

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
            logger.warning("Reward failed: %s", e)

        episode = Episode(
            episode_id=f"fsdp_ep_{step}",
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
            try:
                await weight_sync.push(step + 1)
            except Exception as e:
                logger.warning("[FSDP-Train] Weight sync push failed (step %d): %s", step, e)

        logger.info("[FSDP-Train] Step %d complete: %s", step, result)
        step += 1


async def grpo_fsdp_main(run_id: int = 0):
    """Main entry point for standalone FSDP GRPO training."""
    ensure_ascend_custom_opp_path()

    bridge = create_config_bridge(backend="fsdp")
    forge_cfg, raw_cfg, _ = bridge.parse_and_build(run_id=run_id)

    bridge.setup_name_resolve(raw_cfg)
    bridge.save_metadata(raw_cfg)

    os.makedirs(forge_cfg.resolve_log_dir(), exist_ok=True)
    os.makedirs(forge_cfg.fileroot, exist_ok=True)

    logger.info(
        "GRPO-FSDP: experiment=%s, model=%s, steps=%d, train_ws=%d, gen_ws=%d",
        forge_cfg.experiment_name,
        forge_cfg.model_path,
        forge_cfg.backend_config.get("max_steps", 0),
        forge_cfg.train_world_size,
        forge_cfg.gen_world_size,
    )

    await init_provisioner()

    generator = await Generator.options(
        procs=1, with_gpus=True, mesh_name="generator"
    ).as_actor(engine_args=forge_cfg.engine_args)

    reward = await RewardActor.options(
        procs=1, with_gpus=False, mesh_name="reward"
    ).as_actor()
    if forge_cfg.reward_fn_path:
        await reward.setup.call(forge_cfg.reward_fn_path)

    engine = create_engine(backend="fsdp", config=forge_cfg.backend_config)

    trainer = await TrainerActor.options(
        procs=forge_cfg.train_world_size, with_gpus=True, mesh_name="trainer"
    ).as_actor(engine=engine)

    weight_sync = None
    try:
        weight_sync = await _create_weight_sync(forge_cfg, trainer, generator)
    except Exception as e:
        logger.warning("Weight sync not available (%s), training without live sync", e)

    info_mesh = await trainer.initialize.call()
    _, info = next(iter(info_mesh.items()))
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info("Trainer ready: max_steps=%d, start=%d", max_steps, start_step)

    batch_adapter = create_batch_adapter(backend="fsdp")

    try:
        await _engine_training_loop(
            trainer, generator, reward, batch_adapter, weight_sync,
            max_steps=max_steps, start_step=start_step,
        )
    except Exception as e:
        logger.error("FSDP GRPO training failed: %s", e)
        raise
    finally:
        if weight_sync is not None:
            await weight_sync.shutdown()
        logger.info("Shutting down...")
        await shutdown()

    logger.info("GRPO-FSDP training complete.")


def main():
    from monarch._src.actor.actor_mesh import context

    context()
    asyncio.run(grpo_fsdp_main(run_id=0))


if __name__ == "__main__":
    main()
