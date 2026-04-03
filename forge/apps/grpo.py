"""GRPO training pipeline for Forge.

``run_pipeline()`` is the framework-agnostic training loop. It depends
ONLY on ``RolloutStage``, ``TrainStage``, and ``ReplayBuffer`` — no
Monarch, no AReaL imports, no framework coupling.

``main()`` is the CLI entry point that delegates to the Monarch adapter
for actor creation, then calls ``run_pipeline()``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass

from forge.api.engine import RolloutStage, TrainStage
from forge.core.replay_buffer import ReplayBuffer

logger = logging.getLogger("forge.apps.grpo")


@dataclass
class PipelineConfig:
    """Tunable parameters for the async training pipeline."""

    max_staleness: int = 3
    buffer_max_size: int = 8


async def run_pipeline(
    rollout: RolloutStage,
    train: TrainStage,
    buffer: ReplayBuffer,
    *,
    max_steps: int,
    start_step: int = 0,
    config: PipelineConfig | None = None,
) -> None:
    """Async rollout + training pipeline.

    This is the core loop — transparent and hackable. It depends only
    on Protocol interfaces, not on any distributed framework.

    Copy this function and modify it for custom pipelines.
    """
    if config is None:
        config = PipelineConfig()

    rollout_step = start_step
    train_step = start_step
    rollout_done = asyncio.Event()

    async def _rollout_loop() -> None:
        nonlocal rollout_step
        while rollout_step < max_steps:
            step = rollout_step
            logger.info("[Rollout] Step %d: producing batch", step)
            batch = await rollout.produce_batch(step)
            buffer.add(batch, policy_version=step)
            logger.info("[Rollout] Step %d: buffer_size=%d", step, buffer.size)
            rollout_step += 1
        rollout_done.set()

    async def _training_loop() -> None:
        nonlocal train_step
        while train_step < max_steps:
            batch = buffer.sample(
                current_version=train_step,
                max_staleness=config.max_staleness,
            )
            if batch is None:
                if rollout_done.is_set():
                    logger.warning("[Training] Buffer empty and rollout done")
                    break
                await asyncio.sleep(0.5)
                continue

            step = train_step
            metrics = await train.consume_batch(batch, step)
            logger.info(
                "[Step %d/%d] epoch=%s, epoch_step=%s",
                step + 1,
                max_steps,
                metrics.get("epoch", "?"),
                metrics.get("epoch_step", "?"),
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

    logger.info("ReplayBuffer final stats: %s", buffer.stats())
    logger.info("Training completed successfully.")


# -------------------------------------------------------------------
# CLI entry point — delegates to Monarch adapter, then runs pipeline
# -------------------------------------------------------------------


def main() -> None:
    """CLI entry point: parse config, set up Monarch actors, run pipeline."""
    from monarch._src.actor.actor_mesh import context

    from areal.api.cli_args import parse_cli_args

    context()
    config, _ = parse_cli_args(sys.argv[1:])

    asyncio.run(_main_async(config))


async def _main_async(config) -> None:
    from forge.adapters.monarch.setup import setup_grpo

    ctx = await setup_grpo(config, run_id=0)

    await run_pipeline(
        rollout=ctx["rollout"],
        train=ctx["train"],
        buffer=ctx["buffer"],
        max_steps=ctx["max_steps"],
        start_step=ctx["start_step"],
    )


if __name__ == "__main__":
    main()
