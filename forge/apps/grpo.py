"""GRPO training entry point using Forge actors.

Uses Generator + TrainerActor.train_step for synchronous rollout-and-training.
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
from forge.actors.reward import RewardActor
from forge.actors.trainer import TrainerActor
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.engines import create_config_bridge, create_engine
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("ForgeGRPO")


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

    reward = await RewardActor.options(procs=1, mesh_name="reward").as_actor()
    await reward.setup.call(forge_cfg.reward_fn_path)

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
    ).as_actor(
        backend=backend,
    )

    info_mesh = await trainer.initialize.call()
    _, info = next(iter(info_mesh.items()))
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    logger.info(f"Trainer ready: max_steps={max_steps}, start={start_step}")

    step = start_step
    try:
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
        logger.info("Shutting down all actors...")
        await shutdown()

    logger.info(f"GRPO training complete. Total steps: {step}")


def main():
    from monarch._src.actor.actor_mesh import context

    context()

    asyncio.run(grpo_main(run_id=0))


if __name__ == "__main__":
    main()
