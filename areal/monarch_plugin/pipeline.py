"""Async training pipeline: rollout → replay buffer → training.

Provides a reusable ``run_training_pipeline`` coroutine that runs the
rollout-producer and training-consumer loops concurrently via ``asyncio``.

The pipeline is parameterised by actor references, step range, and staleness
settings.  It does not depend on the launcher or actor specs -- only on the
actor endpoint interfaces.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from areal.utils import logging

logger = logging.getLogger("MonarchPipeline")


@dataclass
class PipelineConfig:
    """Tunable parameters for the training pipeline."""

    max_staleness: int = 3
    """Maximum allowed staleness between rollout step and training step."""

    replay_buffer_max_size: int = 8
    """Maximum batches in the replay buffer."""


async def run_training_pipeline(
    rollout_actor,
    replay_buffer_actor,
    trainer_actor,
    *,
    max_steps: int,
    start_step: int = 0,
    multi_rank: bool = False,
    config: PipelineConfig = PipelineConfig(),
) -> None:
    """Run the rollout + training pipeline until all steps complete.

    Parameters
    ----------
    rollout_actor
        Actor with ``do_rollout(step) -> batch_data`` endpoint.
    replay_buffer_actor
        Actor with ``add_batch(batch, step)``, ``sample_batch(step, staleness)``,
        and ``buffer_size()`` endpoints.
    trainer_actor
        Actor with ``train_on_batch(batch, step) -> result`` endpoint.
    max_steps
        Total number of training steps to run.
    start_step
        Step to resume from (for recovery).
    multi_rank
        If ``True``, trainer_actor has multiple ranks and ``call()`` (broadcast)
        should be used instead of ``call_one()``.
    config
        Pipeline tunables (staleness, buffer size).
    """
    rollout_step = start_step
    train_step_counter = start_step
    rollout_done = asyncio.Event()

    async def _rollout_loop():
        """Produce rollout batches via independent RolloutActor."""
        nonlocal rollout_step
        while rollout_step < max_steps:
            step = rollout_step
            logger.info(
                f"[Rollout] Starting rollout for step {step} "
                f"(on independent RolloutActor)"
            )
            batch_data = await rollout_actor.do_rollout.call_one(step)
            await replay_buffer_actor.add_batch.call_one(batch_data, step)
            buf_size = await replay_buffer_actor.buffer_size.call_one()
            logger.info(
                f"[Rollout] Step {step} batch added to buffer (buffer_size={buf_size})"
            )
            rollout_step += 1
        rollout_done.set()

    async def _training_loop():
        """Consume batches from ReplayBuffer and train."""
        nonlocal train_step_counter
        while train_step_counter < max_steps:
            batch = await replay_buffer_actor.sample_batch.call_one(
                train_step_counter, config.max_staleness
            )
            if batch is None:
                if rollout_done.is_set():
                    logger.warning(
                        "[Training] Buffer empty and rollout done, stopping training"
                    )
                    break
                await asyncio.sleep(0.5)
                continue

            step = train_step_counter
            if multi_rank:
                result_mesh = await trainer_actor.train_on_batch.call(batch, step)
                result = result_mesh.item(npu=0)
            else:
                result = await trainer_actor.train_on_batch.call_one(batch, step)
            logger.info(
                f"[Step {step + 1}/{max_steps}] "
                f"epoch={result['epoch']}, "
                f"epoch_step={result['epoch_step']}"
            )
            train_step_counter += 1

    rollout_task = asyncio.create_task(_rollout_loop())
    train_task = asyncio.create_task(_training_loop())

    await asyncio.gather(rollout_task, train_task)

    buf_stats = await replay_buffer_actor.get_stats.call_one()
    logger.info(f"ReplayBuffer final stats: {buf_stats}")
    logger.info("Training completed successfully.")
