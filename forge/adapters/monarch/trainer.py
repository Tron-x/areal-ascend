"""FSDP training adapter for Forge on Monarch.

Wraps AReaL's PPOTrainer and FSDPEngine to implement the Forge
``TrainEngine`` protocol. Ported from ``areal/monarch_plugin/trainer_actor.py``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, NamedTuple

from monarch.actor import endpoint

from forge.adapters.monarch.actor import MonarchForgeActor

logger = logging.getLogger("forge.monarch.trainer")


class _StepContext(NamedTuple):
    trainer: object
    config: object
    step_args: dict
    global_step: int
    epoch: int
    step_in_epoch: int


class FSDPTrainerActor(MonarchForgeActor):
    """Monarch actor wrapping AReaL's PPOTrainer + FSDPEngine.

    Implements the complete training lifecycle: initialization,
    rollout, training steps, weight sync, and checkpointing.
    """

    procs = 1
    with_gpus = True

    def __init__(
        self,
        cli_args: list,
        env_vars: dict,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        generator_actor,
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
        self._xccl_weight_update_alloc_mode = xccl_weight_update_alloc_mode
        self._trainer = None
        self._train_kwargs: dict = {}
        self._max_steps = 0
        self._steps_per_epoch = 1

    @endpoint
    def initialize(self) -> dict:
        if self._rank < 0:
            from monarch._src.actor.actor_mesh import current_rank

            self._rank = current_rank().rank
            logger.info("Auto-detected rank=%d", self._rank)

        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ.update(self._env_vars)

        if self._world_size > 1:
            import torch

            torch.npu.set_device(self._rank)

        from areal import PPOTrainer
        from areal.monarch_plugin.monarch_inf_engine import MonarchVLLMEngine

        generator_ref = self._generator
        reward_ref = self._reward
        agent_ref = self._agent

        _captured: dict = {}
        _orig_train = PPOTrainer.train
        _orig_exit = PPOTrainer.__exit__
        _orig_init_rollout = PPOTrainer._init_rollout

        def _intercept_train(trainer_self, **kwargs):
            _captured["trainer"] = trainer_self
            _captured["kwargs"] = kwargs

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

            engine = MonarchVLLMEngine(
                rollout_config,
                generator_ref,
                reward_actor=reward_ref,
                agent_actor=agent_ref,
            )
            engine.initialize(
                train_data_parallel_size=trainer_self.allocation_mode.train.dp_size,
            )
            return engine

        PPOTrainer.train = _intercept_train
        PPOTrainer.__exit__ = _intercept_exit
        PPOTrainer._init_rollout = _patched_init_rollout

        from areal.engine.fsdp_engine import FSDPEngine

        _orig_connect = FSDPEngine.connect_engine

        def _patched_connect(engine_self, engine, meta):
            if (
                meta.type == "xccl"
                and meta.alloc_mode is not None
                and self._xccl_weight_update_alloc_mode is not None
            ):
                meta.alloc_mode = self._xccl_weight_update_alloc_mode
            return _orig_connect(engine_self, engine, meta)

        FSDPEngine.connect_engine = _patched_connect

        try:
            script_path = self._cli_args[0]
            spec = importlib.util.spec_from_file_location("_experiment", script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main(self._cli_args[1:])
        except Exception as e:
            import traceback

            logger.error("Script execution failed: %s", e)
            traceback.print_exc()
            raise RuntimeError(f"Trainer initialization failed: {e}") from e
        finally:
            PPOTrainer.train = _orig_train
            PPOTrainer.__exit__ = _orig_exit
            PPOTrainer._init_rollout = _orig_init_rollout
            FSDPEngine.connect_engine = _orig_connect

        if "trainer" not in _captured:
            raise RuntimeError("Could not capture PPOTrainer instance")

        self._trainer = _captured["trainer"]
        self._train_kwargs = _captured["kwargs"]

        if (
            hasattr(self._trainer, "weight_update_meta")
            and self._trainer.weight_update_meta.alloc_mode is not None
            and self._xccl_weight_update_alloc_mode is not None
        ):
            self._trainer.weight_update_meta.alloc_mode = (
                self._xccl_weight_update_alloc_mode
            )

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

        return {
            "status": "ready",
            "rank": self._rank,
            "max_steps": self._max_steps,
            "start_step": start_step,
            "steps_per_epoch": self._steps_per_epoch,
        }

    @endpoint
    def train_step(self, global_step: int) -> dict:
        batch_data = self._do_rollout_impl(global_step)
        return self._train_on_batch_impl(batch_data, global_step)

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        return self._do_rollout_impl(global_step)

    @endpoint
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        return self._train_on_batch_impl(batch_data, global_step)

    def _do_rollout_impl(self, global_step: int) -> dict:
        import numpy as np
        import torch

        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = self._trainer
        step_in_epoch = global_step % self._steps_per_epoch

        sa = {"global_step": global_step, "epoch_step": step_in_epoch}
        workflow = self._train_kwargs.get("workflow")
        workflow_kwargs = self._train_kwargs.get("workflow_kwargs")
        dynamic_filter_fn = self._train_kwargs.get("dynamic_filter_fn")

        with (
            stats_tracker.record_timing("rollout"),
            perf_tracer.trace_scope(
                "train.rollout", category=Category.COMPUTE, args=sa
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

    def _train_on_batch_impl(self, batch_data: dict, global_step: int) -> dict:
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

        rollout_batch = self._ensure_batch_on_device(batch_data)
        self._compute_logps(ctx, rollout_batch)
        adv_batch = self._compute_advantages(ctx, rollout_batch)
        trainer.saver.maybe_wait_for_staging()
        self._ppo_update(ctx, adv_batch)
        self._critic_update(ctx, adv_batch)
        self._sync_weights(ctx)

        eval_workflow = self._train_kwargs.get("eval_workflow")
        eval_workflow_kwargs = self._train_kwargs.get("eval_workflow_kwargs")
        self._save_and_evaluate(
            ctx, rollout_batch, adv_batch, eval_workflow, eval_workflow_kwargs
        )
        self._log_and_resume(ctx)

        return {"global_step": global_step, "epoch": epoch, "epoch_step": step_in_epoch}

    def _ensure_batch_on_device(self, batch_data: Any) -> dict:
        import pickle

        import torch

        if isinstance(batch_data, bytes):
            try:
                batch = pickle.loads(batch_data)
                return self._ensure_batch_on_device(batch)
            except Exception as e:
                raise RuntimeError(f"Failed to unpickle batch: {e}") from e

        if not isinstance(batch_data, dict) or not batch_data:
            return batch_data

        first_val = next(iter(batch_data.values()))
        if isinstance(first_val, list):
            return self._deserialise_batch(batch_data)
        if isinstance(first_val, torch.Tensor):
            device = next(self._trainer.actor.model.parameters()).device
            return {
                k: v.to(device=device)
                if isinstance(v, torch.Tensor) and v.device != device
                else v
                for k, v in batch_data.items()
            }
        return batch_data

    def _deserialise_batch(self, batch_data: dict) -> dict:
        import torch

        device = next(self._trainer.actor.model.parameters()).device
        restored = {}
        for k, v in batch_data.items():
            if isinstance(v, list):
                try:
                    t = torch.tensor(v)
                    t = t.to(
                        dtype=torch.float32 if t.is_floating_point() else t.dtype,
                        device=device,
                    )
                    restored[k] = t
                except (ValueError, TypeError):
                    restored[k] = v
            else:
                restored[k] = v
        return restored

    def _compute_logps(self, ctx: _StepContext, rollout_batch: dict) -> None:
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
                rollout_batch["values"] = trainer.critic.compute_values(rollout_batch)

        if ctx.config.actor.should_compute_prox_logp():
            with (
                stats_tracker.record_timing("recompute_logp"),
                perf_tracer.trace_scope(
                    "train.recompute_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                rollout_batch["prox_logp"] = trainer.actor.compute_logp(rollout_batch)

        if trainer.ref is not None:
            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                rollout_batch["ref_logp"] = trainer.ref.compute_logp(rollout_batch)

        if trainer.teacher is not None:
            with (
                stats_tracker.record_timing("teacher_logp"),
                perf_tracer.trace_scope(
                    "train.teacher_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                rollout_batch["teacher_logp"] = trainer.teacher.compute_logp(
                    rollout_batch
                )
                rollout_batch["rl_loss_weight"] = ctx.config.teacher.rl_loss_weight
                rollout_batch["distill_loss_weight"] = (
                    ctx.config.teacher.distill_loss_weight
                )

    def _compute_advantages(self, ctx: _StepContext, rollout_batch: dict) -> dict:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        with (
            stats_tracker.record_timing("compute_advantage"),
            perf_tracer.trace_scope(
                "train.compute_advantage", category=Category.COMPUTE, args=ctx.step_args
            ),
        ):
            return ctx.trainer.actor.compute_advantages(rollout_batch)

    def _ppo_update(self, ctx: _StepContext, adv_batch: dict) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        with (
            stats_tracker.record_timing("train_step"),
            perf_tracer.trace_scope(
                "train.ppo_update", category=Category.COMPUTE, args=ctx.step_args
            ),
        ):
            ctx.trainer.actor.ppo_update(adv_batch)
            ctx.trainer.actor.step_lr_scheduler()

    def _critic_update(self, ctx: _StepContext, adv_batch: dict) -> None:
        if ctx.trainer.critic is None:
            return
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        with (
            stats_tracker.record_timing("critic_train_step"),
            perf_tracer.trace_scope(
                "train.critic_ppo_update", category=Category.COMPUTE, args=ctx.step_args
            ),
        ):
            ctx.trainer.critic.ppo_update(adv_batch)
            ctx.trainer.critic.step_lr_scheduler()

    def _sync_weights(self, ctx: _StepContext) -> None:
        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        trainer = ctx.trainer
        trainer.rollout.pause()
        with (
            stats_tracker.record_timing("update_weights"),
            perf_tracer.trace_scope(
                "train.update_weights", category=Category.COMM, args=ctx.step_args
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
        self, ctx, rollout_batch, adv_batch, eval_workflow, eval_workflow_kwargs
    ):
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
                eval_workflow_kwargs=eval_workflow_kwargs,
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
            trainer.actor.clear_batches(rollout_batch, adv_batch)

    def _log_and_resume(self, ctx: _StepContext) -> None:
        from areal.utils import perf_tracer

        trainer = ctx.trainer
        with perf_tracer.trace_scope(
            "train.log_stats", category=perf_tracer.Category.INSTR, args=ctx.step_args
        ):
            trainer._export_and_commit_stats(
                epoch=ctx.epoch,
                epoch_step=ctx.step_in_epoch,
                global_step=ctx.global_step,
            )
        trainer.rollout.resume()
        trainer._save_perf_tracer(step=ctx.global_step)

    @endpoint
    def shutdown(self) -> None:
        if self._trainer is not None:
            logger.info("FSDPTrainerActor[rank=%d] shutting down", self._rank)
            try:
                self._trainer.close()
            except Exception as e:
                logger.warning("Error during trainer close: %s", e)
            self._trainer = None
