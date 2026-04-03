"""AReaL training backend — wraps PPOTrainer + FSDPEngine for Forge.

Implements ``forge.core.protocols.TrainBackend``.

All AReaL-specific logic (monkey-patching PPOTrainer.train / _init_rollout,
FSDPEngine.connect_engine, perf_tracer, stats_tracker) is confined here.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class _StepContext(NamedTuple):
    trainer: object
    config: object
    step_args: dict
    global_step: int
    epoch: int
    step_in_epoch: int


class AReaLTrainBackend:
    """Training backend that wraps AReaL's PPOTrainer and FSDPEngine.

    Satisfies the ``TrainBackend`` protocol.

    Lifecycle::

        backend = AReaLTrainBackend(cli_args=[...], env_vars={...}, ...)
        info = backend.initialize()
        result = backend.train_step(step)
        backend.shutdown()
    """

    def __init__(
        self,
        cli_args: list,
        env_vars: dict,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        generator_actor=None,
        reward_actor=None,
        agent_actor=None,
        xccl_weight_update_alloc_mode=None,
    ):
        self._cli_args = cli_args
        self._env_vars = env_vars
        self._rank = rank
        self._world_size = world_size
        self._master_addr = master_addr
        self._master_port = master_port
        self._generator = generator_actor
        self._reward = reward_actor
        self._agent = agent_actor
        self._xccl_alloc_mode = xccl_weight_update_alloc_mode
        self._trainer = None
        self._train_kwargs: dict = {}
        self._max_steps = 0
        self._steps_per_epoch = 1

    def initialize(self) -> dict:
        if self._rank < 0:
            from monarch._src.actor.actor_mesh import current_rank

            self._rank = current_rank().rank

        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ.update(self._env_vars)

        if self._world_size > 1:
            import torch

            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.set_device(self._rank)
            elif torch.cuda.is_available():
                torch.cuda.set_device(self._rank)

        logger.info(
            f"AReaLTrainBackend[rank={self._rank}] initialising: "
            f"WORLD_SIZE={self._world_size}"
        )

        self._trainer, self._train_kwargs = self._capture_trainer()

        config = self._trainer.config
        total_epochs = (
            self._train_kwargs.get("total_epochs") or config.total_train_epochs
        )
        self._steps_per_epoch = len(self._trainer.train_dataloader)
        self._max_steps = total_epochs * self._steps_per_epoch
        if config.total_train_steps is not None:
            self._max_steps = min(self._max_steps, config.total_train_steps)

        start_step = 0
        if self._trainer.recover_info is not None:
            start_step = self._trainer.recover_info.last_step_info.next().global_step

        logger.info(
            f"AReaLTrainBackend[rank={self._rank}] ready: "
            f"max_steps={self._max_steps}, start_step={start_step}"
        )
        return {
            "status": "ready",
            "rank": self._rank,
            "max_steps": self._max_steps,
            "start_step": start_step,
            "steps_per_epoch": self._steps_per_epoch,
        }

    def _capture_trainer(self) -> tuple:
        """Load the experiment script and capture the PPOTrainer instance.

        Uses targeted patching: only intercepts ``train()`` to capture the
        trainer reference, and ``_init_rollout`` to inject the inference bridge.
        """
        from areal import PPOTrainer
        from areal.engine.fsdp_engine import FSDPEngine

        generator_ref = self._generator
        reward_ref = self._reward
        agent_ref = self._agent
        xccl_alloc = self._xccl_alloc_mode

        captured: dict = {}
        _orig_train = PPOTrainer.train
        _orig_exit = PPOTrainer.__exit__
        _orig_init_rollout = PPOTrainer._init_rollout
        _orig_connect = FSDPEngine.connect_engine

        def _intercept_train(trainer_self, **kwargs):
            captured["trainer"] = trainer_self
            captured["kwargs"] = kwargs

        def _intercept_exit(trainer_self, exc_type, exc_value, traceback):
            pass

        def _patched_init_rollout(
            trainer_self, rollout_config, is_eval=False, lora_path=None
        ):
            from areal.utils.environ import is_single_controller

            if is_single_controller():
                return _orig_init_rollout(
                    trainer_self, rollout_config, is_eval, lora_path
                )

            from forge.adapters.areal.inference_bridge import AReaLInferenceBridge

            engine = AReaLInferenceBridge(
                rollout_config,
                generator_ref,
                reward_actor=reward_ref,
                agent_actor=agent_ref,
            )
            engine.initialize(
                train_data_parallel_size=trainer_self.allocation_mode.train.dp_size,
            )
            return engine

        def _patched_connect(engine_self, engine, meta):
            if (
                meta.type == "xccl"
                and meta.alloc_mode is not None
                and xccl_alloc is not None
            ):
                meta.alloc_mode = xccl_alloc
            return _orig_connect(engine_self, engine, meta)

        PPOTrainer.train = _intercept_train
        PPOTrainer.__exit__ = _intercept_exit
        PPOTrainer._init_rollout = _patched_init_rollout
        FSDPEngine.connect_engine = _patched_connect

        try:
            script_path = self._cli_args[0]
            logger.info(f"Loading script: {script_path}")
            spec = importlib.util.spec_from_file_location(
                "_monarch_experiment", script_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main(self._cli_args[1:])
        except Exception as e:
            import traceback

            logger.error(f"Script execution failed: {e}")
            traceback.print_exc()
            raise RuntimeError(f"AReaLTrainBackend initialization failed: {e}") from e
        finally:
            PPOTrainer.train = _orig_train
            PPOTrainer.__exit__ = _orig_exit
            PPOTrainer._init_rollout = _orig_init_rollout
            FSDPEngine.connect_engine = _orig_connect

        if "trainer" not in captured:
            raise RuntimeError(
                f"Could not capture PPOTrainer from {self._cli_args[0]}."
            )

        trainer = captured["trainer"]

        if (
            hasattr(trainer, "weight_update_meta")
            and trainer.weight_update_meta.alloc_mode is not None
            and xccl_alloc is not None
        ):
            trainer.weight_update_meta.alloc_mode = xccl_alloc

        return trainer, captured["kwargs"]

    # ------------------------------------------------------------------
    # TrainBackend protocol methods
    # ------------------------------------------------------------------

    def train_step(self, global_step: int) -> dict:
        batch_data = self.do_rollout(global_step)
        return self.train_on_batch(batch_data, global_step)

    def do_rollout(self, global_step: int) -> dict:
        import numpy as np
        import torch

        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = self._trainer
        step_in_epoch = global_step % self._steps_per_epoch

        ctx = _StepContext(
            trainer=trainer,
            config=trainer.config,
            step_args={"global_step": global_step, "epoch_step": step_in_epoch},
            global_step=global_step,
            epoch=global_step // self._steps_per_epoch,
            step_in_epoch=step_in_epoch,
        )

        workflow = self._train_kwargs.get("workflow")
        workflow_kwargs = self._train_kwargs.get("workflow_kwargs")
        dynamic_filter_fn = self._train_kwargs.get("dynamic_filter_fn")

        with (
            stats_tracker.record_timing("rollout"),
            perf_tracer.trace_scope(
                "train.rollout", category=Category.COMPUTE, args=ctx.step_args
            ),
        ):
            rollout_batch = trainer.actor.prepare_batch(
                trainer.train_dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=dynamic_filter_fn,
                group_size=trainer.config.gconfig.n_samples,
                dynamic_bs=trainer.config.dynamic_bs,
            )

        serialised = {}
        for k, v in rollout_batch.items():
            if isinstance(v, torch.Tensor):
                serialised[k] = v.cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                serialised[k] = v.tolist()
            else:
                serialised[k] = v
        return serialised

    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        trainer = self._trainer
        epoch = global_step // self._steps_per_epoch
        step_in_epoch = global_step % self._steps_per_epoch

        ctx = _StepContext(
            trainer=trainer,
            config=trainer.config,
            step_args={"global_step": global_step, "epoch_step": step_in_epoch},
            global_step=global_step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )

        rollout_batch = _batch_to_device(batch_data, trainer)

        self._compute_logps(ctx, rollout_batch)
        adv_batch = self._compute_advantages(ctx, rollout_batch)

        trainer.saver.maybe_wait_for_staging()

        self._ppo_update(ctx, adv_batch)
        self._critic_update(ctx, adv_batch)
        self._sync_weights(ctx)

        eval_workflow = self._train_kwargs.get("eval_workflow")
        eval_wf_kwargs = self._train_kwargs.get("eval_workflow_kwargs")
        self._save_and_evaluate(
            ctx, rollout_batch, adv_batch, eval_workflow, eval_wf_kwargs
        )
        self._log_and_resume(ctx)

        return {
            "global_step": global_step,
            "epoch": epoch,
            "epoch_step": step_in_epoch,
        }

    def train_on_buffered_batch(
        self, batch_data: dict, global_step: int, skip_weight_sync: bool = False
    ) -> dict:
        """Train on a batch from ReplayBuffer, optionally deferring weight sync.

        When ``skip_weight_sync=True``, the caller is responsible for invoking
        ``sync_weights()`` separately.  This enables the async pipeline to
        overlap the next rollout with the weight transfer.
        """

        trainer = self._trainer
        epoch = global_step // self._steps_per_epoch
        step_in_epoch = global_step % self._steps_per_epoch

        ctx = _StepContext(
            trainer=trainer,
            config=trainer.config,
            step_args={"global_step": global_step, "epoch_step": step_in_epoch},
            global_step=global_step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )

        rollout_batch = _batch_to_device(batch_data, trainer)

        self._compute_logps(ctx, rollout_batch)
        adv_batch = self._compute_advantages(ctx, rollout_batch)

        trainer.saver.maybe_wait_for_staging()

        self._ppo_update(ctx, adv_batch)
        self._critic_update(ctx, adv_batch)

        if not skip_weight_sync:
            self._sync_weights(ctx)

        eval_workflow = self._train_kwargs.get("eval_workflow")
        eval_wf_kwargs = self._train_kwargs.get("eval_workflow_kwargs")
        self._save_and_evaluate(
            ctx, rollout_batch, adv_batch, eval_workflow, eval_wf_kwargs
        )
        self._log_and_resume(ctx)

        return {
            "global_step": global_step,
            "epoch": epoch,
            "epoch_step": step_in_epoch,
        }

    def sync_weights(self, global_step: int) -> dict:
        """Push updated weights to Generator, standalone from training.

        Called by the orchestrator after ``train_on_buffered_batch``
        when weight sync is deferred.
        """
        epoch = global_step // self._steps_per_epoch
        step_in_epoch = global_step % self._steps_per_epoch

        ctx = _StepContext(
            trainer=self._trainer,
            config=self._trainer.config,
            step_args={"global_step": global_step, "epoch_step": step_in_epoch},
            global_step=global_step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )
        self._sync_weights(ctx)
        return {"synced_version": global_step + 1}

    def get_train_metadata(self) -> dict:
        """Return training metadata for the orchestrator."""
        return {
            "max_steps": self._max_steps,
            "steps_per_epoch": self._steps_per_epoch,
            "rank": self._rank,
            "world_size": self._world_size,
        }

    def shutdown(self) -> None:
        if self._trainer is not None:
            logger.info(f"AReaLTrainBackend[rank={self._rank}] shutting down")
            try:
                self._trainer.close()
            except Exception as e:
                logger.warning(f"Error during trainer close: {e}")
            self._trainer = None

    # ------------------------------------------------------------------
    # Internal training stages (AReaL-specific)
    # ------------------------------------------------------------------

    def _compute_logps(self, ctx: _StepContext, batch: dict) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        sa = ctx.step_args

        if trainer.critic is not None:
            with (
                stats_tracker.record_timing("critic_values"),
                perf_tracer.trace_scope(
                    "train.compute_values", category=Category.COMPUTE, args=sa
                ),
            ):
                batch["values"] = trainer.critic.compute_values(batch)

        if ctx.config.actor.should_compute_prox_logp():
            with (
                stats_tracker.record_timing("recompute_logp"),
                perf_tracer.trace_scope(
                    "train.recompute_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                batch["prox_logp"] = trainer.actor.compute_logp(batch)

        if trainer.ref is not None:
            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                batch["ref_logp"] = trainer.ref.compute_logp(batch)

        if trainer.teacher is not None:
            with (
                stats_tracker.record_timing("teacher_logp"),
                perf_tracer.trace_scope(
                    "train.teacher_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                batch["teacher_logp"] = trainer.teacher.compute_logp(batch)
                batch["rl_loss_weight"] = ctx.config.teacher.rl_loss_weight
                batch["distill_loss_weight"] = ctx.config.teacher.distill_loss_weight

    def _compute_advantages(self, ctx: _StepContext, batch: dict) -> dict:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        with (
            stats_tracker.record_timing("compute_advantage"),
            perf_tracer.trace_scope(
                "train.compute_advantage",
                category=Category.COMPUTE,
                args=ctx.step_args,
            ),
        ):
            return trainer.actor.compute_advantages(batch)

    def _ppo_update(self, ctx: _StepContext, adv_batch: dict) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        with (
            stats_tracker.record_timing("train_step"),
            perf_tracer.trace_scope(
                "train.ppo_update",
                category=Category.COMPUTE,
                args=ctx.step_args,
            ),
        ):
            trainer.actor.ppo_update(adv_batch)
            trainer.actor.step_lr_scheduler()

    def _critic_update(self, ctx: _StepContext, adv_batch: dict) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        if trainer.critic is None:
            return
        with (
            stats_tracker.record_timing("critic_train_step"),
            perf_tracer.trace_scope(
                "train.critic_ppo_update",
                category=Category.COMPUTE,
                args=ctx.step_args,
            ),
        ):
            trainer.critic.ppo_update(adv_batch)
            trainer.critic.step_lr_scheduler()

    def _sync_weights(self, ctx: _StepContext) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        trainer.rollout.pause()
        with (
            stats_tracker.record_timing("update_weights"),
            perf_tracer.trace_scope(
                "train.update_weights",
                category=Category.COMM,
                args=ctx.step_args,
            ),
        ):
            new_version = ctx.global_step + 1
            versioned_meta = trainer.weight_update_meta.with_version(new_version)
            trainer.actor.update_weights(versioned_meta)
            trainer.actor.set_version(new_version)
            if trainer.critic is not None:
                trainer.critic.set_version(new_version)
            trainer.rollout.set_version(new_version)
            if trainer.eval_rollout is not None:
                trainer.eval_rollout.set_version(new_version)

    def _save_and_evaluate(
        self, ctx, batch, adv_batch, eval_workflow, eval_wf_kwargs
    ) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        sa = ctx.step_args

        with (
            stats_tracker.record_timing("save"),
            perf_tracer.trace_scope("train.save", category=Category.IO, args=sa),
        ):
            trainer._save_hf(
                epoch=ctx.epoch,
                epoch_step=ctx.step_in_epoch,
                global_step=ctx.global_step,
            )

        with (
            stats_tracker.record_timing("checkpoint_for_recover"),
            perf_tracer.trace_scope("train.checkpoint", category=Category.IO, args=sa),
        ):
            trainer._save_recover_checkpoint(
                epoch=ctx.epoch,
                epoch_step=ctx.step_in_epoch,
                global_step=ctx.global_step,
            )

        with (
            stats_tracker.record_timing("eval"),
            perf_tracer.trace_scope("train.eval", category=Category.COMPUTE, args=sa),
        ):
            trainer._evaluate(
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_wf_kwargs,
                epoch=ctx.epoch,
                epoch_step=ctx.step_in_epoch,
                global_step=ctx.global_step,
            )

        with (
            stats_tracker.record_timing("clear_batches"),
            perf_tracer.trace_scope(
                "train.clear_batches", category=Category.INSTR, args=sa
            ),
        ):
            trainer.actor.clear_batches(batch, adv_batch)

    def _log_and_resume(self, ctx: _StepContext) -> None:
        from areal.utils import perf_tracer
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        with perf_tracer.trace_scope(
            "train.log_stats", category=Category.INSTR, args=ctx.step_args
        ):
            trainer._export_and_commit_stats(
                epoch=ctx.epoch,
                epoch_step=ctx.step_in_epoch,
                global_step=ctx.global_step,
            )
        trainer.rollout.resume()
        trainer._save_perf_tracer(step=ctx.global_step)


def _batch_to_device(batch_data: Any, trainer) -> dict:
    """Move a batch (possibly serialized) to the trainer's device."""
    import pickle

    import torch

    if isinstance(batch_data, bytes):
        return _batch_to_device(pickle.loads(batch_data), trainer)

    if not isinstance(batch_data, dict) or not batch_data:
        return batch_data

    device = next(trainer.actor.model.parameters()).device

    first_val = next(iter(batch_data.values()))
    if isinstance(first_val, list):
        restored = {}
        for k, v in batch_data.items():
            if isinstance(v, list):
                try:
                    t = torch.tensor(v)
                    if t.is_floating_point():
                        t = t.to(dtype=torch.float32, device=device)
                    else:
                        t = t.to(device=device)
                    restored[k] = t
                except (ValueError, TypeError):
                    restored[k] = v
            else:
                restored[k] = v
        return restored

    if isinstance(first_val, torch.Tensor):
        return {
            k: v.to(device=device)
            if isinstance(v, torch.Tensor) and v.device != device
            else v
            for k, v in batch_data.items()
        }

    return batch_data
