"""In-process FSDP training actor for Monarch orchestration.

TrainerActor runs the full AReaL training pipeline (PPOTrainer) inside a
Monarch ProcMesh.  It monkey-patches PPOTrainer and FSDPEngine at runtime
to wire in Monarch-based inference (MonarchVLLMEngine) and fix XCCL weight
sync alloc_mode.
"""

from __future__ import annotations

import importlib.util
import os
from typing import NamedTuple

from monarch.actor import Actor, endpoint

from areal.api import AllocationMode
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.perf_tracer import Category

logger = logging.getLogger("TrainerActor")


# ---------------------------------------------------------------------------
# Step context — bundles values shared across all training stages
# ---------------------------------------------------------------------------


class _StepContext(NamedTuple):
    """Immutable bundle passed to every stage method."""

    trainer: object
    config: object
    step_args: dict
    global_step: int
    epoch: int
    step_in_epoch: int


# ---------------------------------------------------------------------------
# TrainerActor
# ---------------------------------------------------------------------------


class TrainerActor(Actor):
    """In-process FSDP training actor with Monarch RPC rollout and reward.

    Accepts ``generator_actor``, ``reward_actor``, and ``agent_actor``
    references and monkey-patches ``PPOTrainer._init_rollout`` to create
    ``MonarchVLLMEngine`` with all actor references.

    ``xccl_weight_update_alloc_mode``: ``AllocationMode`` used when patching
    FSDPEngine XCCL connect (see ``weight_sync.resolve_xccl_alloc_mode``).
    """

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
        xccl_weight_update_alloc_mode: AllocationMode | None = None,
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

            rank_point = current_rank()
            self._rank = rank_point.rank
            logger.info(
                f"TrainerActor auto-detected rank={self._rank} from ProcMesh coordinate"
            )

        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ.update(self._env_vars)

        if self._world_size > 1:
            import torch

            torch.npu.set_device(self._rank)

        logger.info(
            f"TrainerActor[rank={self._rank}] initialising: "
            f"WORLD_SIZE={self._world_size}, "
            f"ASCEND_RT_VISIBLE_DEVICES="
            f"{os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'N/A')}"
        )

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
            """Replace RemotevLLMEngine with MonarchVLLMEngine."""
            from areal.utils.environ import is_single_controller

            if is_single_controller():
                return _orig_init_rollout(
                    trainer_self, rollout_config, is_eval, lora_path
                )

            config = rollout_config
            engine = MonarchVLLMEngine(
                config,
                generator_ref,
                reward_actor=reward_ref,
                agent_actor=agent_ref,
            )
            engine.initialize(
                train_data_parallel_size=trainer_self.allocation_mode.train.dp_size,
            )
            logger.info(
                f"TrainerActor: created MonarchVLLMEngine "
                f"(is_eval={is_eval}, reward_actor={'yes' if reward_ref else 'no'}, "
                f"agent_actor={'yes' if agent_ref else 'no'})"
            )
            return engine

        PPOTrainer.train = _intercept_train
        PPOTrainer.__exit__ = _intercept_exit
        PPOTrainer._init_rollout = _patched_init_rollout

        from areal.engine.fsdp_engine import FSDPEngine

        _orig_connect = FSDPEngine.connect_engine
        ws = self._world_size

        def _patched_connect(engine_self, engine, meta):
            """Fix XCCL weight update alloc_mode for Monarch (see weight_sync)."""
            if (
                meta.type == "xccl"
                and meta.alloc_mode is not None
                and self._xccl_weight_update_alloc_mode is not None
            ):
                orig_gen_ws = meta.alloc_mode.gen.world_size
                fixed = self._xccl_weight_update_alloc_mode
                meta.alloc_mode = fixed
                logger.info(
                    "TrainerActor: XCCL weight-update alloc_mode set for Monarch: "
                    f"gen.world_size {orig_gen_ws} -> {fixed.gen.world_size} "
                    f"(train_world_size={ws})"
                )
            return _orig_connect(engine_self, engine, meta)

        FSDPEngine.connect_engine = _patched_connect

        try:
            script_path = self._cli_args[0]
            spec = importlib.util.spec_from_file_location(
                "_monarch_experiment", script_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main(self._cli_args[1:])
        finally:
            PPOTrainer.train = _orig_train
            PPOTrainer.__exit__ = _orig_exit
            PPOTrainer._init_rollout = _orig_init_rollout
            FSDPEngine.connect_engine = _orig_connect

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

        logger.info(
            f"TrainerActor[rank={self._rank}] ready: "
            f"max_steps={self._max_steps}, start_step={start_step}"
        )
        return {
            "status": "ready",
            "rank": self._rank,
            "max_steps": self._max_steps,
            "start_step": start_step,
            "steps_per_epoch": self._steps_per_epoch,
        }

    # ------------------------------------------------------------------
    # Phase 5 (legacy): combined rollout + training in one endpoint
    # ------------------------------------------------------------------

    @endpoint
    def train_step(self, global_step: int) -> dict:
        """Run one GRPO/PPO training iteration (rollout + train combined)."""
        batch_data = self._do_rollout_impl(global_step)
        return self._train_on_batch_impl(batch_data, global_step)

    # ------------------------------------------------------------------
    # Phase 6: split endpoints for async rollout / training pipeline
    # ------------------------------------------------------------------

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        """Run rollout only, serialise the batch for ReplayBuffer."""
        return self._do_rollout_impl(global_step)

    @endpoint
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        """Train on a pre-produced rollout batch from the ReplayBuffer."""
        return self._train_on_batch_impl(batch_data, global_step)

    # ------------------------------------------------------------------
    # Rollout implementation
    # ------------------------------------------------------------------

    def _do_rollout_impl(self, global_step: int) -> dict:
        """Produce a rollout batch and serialise it."""
        import numpy as np
        import torch

        trainer = self._trainer
        config = trainer.config
        step_in_epoch = global_step % self._steps_per_epoch

        ctx = _StepContext(
            trainer=trainer,
            config=config,
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
                group_size=config.gconfig.n_samples,
                dynamic_bs=config.dynamic_bs,
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

    # ------------------------------------------------------------------
    # Training step orchestrator + stage methods
    # ------------------------------------------------------------------

    def _train_on_batch_impl(self, batch_data: dict, global_step: int) -> dict:
        """Run one training step on a (possibly deserialised) rollout batch.

        Thin orchestrator that delegates to stage methods in sequence.
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

        # 1. Deserialise batch onto device
        rollout_batch = self._ensure_batch_on_device(batch_data)

        # 2. Forward passes: critic values + all log-probabilities
        self._compute_logps(ctx, rollout_batch)

        # 3. Advantage estimation
        adv_batch = self._compute_advantages(ctx, rollout_batch)

        # 4. Wait for async staging before mutating model
        trainer.saver.maybe_wait_for_staging()

        # 5. PPO updates (actor + optional critic)
        self._ppo_update(ctx, adv_batch)
        self._critic_update(ctx, adv_batch)

        # 6. Sync updated weights to inference engines
        self._sync_weights(ctx)

        # 7. Save checkpoints, evaluate, clear batches
        eval_workflow = self._train_kwargs.get("eval_workflow")
        eval_workflow_kwargs = self._train_kwargs.get("eval_workflow_kwargs")
        self._save_and_evaluate(
            ctx, rollout_batch, adv_batch, eval_workflow, eval_workflow_kwargs
        )

        # 8. Log stats and resume rollout
        self._log_and_resume(ctx)

        return {
            "global_step": global_step,
            "epoch": epoch,
            "epoch_step": step_in_epoch,
        }

    # -- Stage 1: batch deserialisation ------------------------------------

    def _ensure_batch_on_device(self, batch_data: dict) -> dict:
        """Deserialise a batch if needed; pass through if already on device."""
        if (
            isinstance(batch_data, dict)
            and batch_data
            and isinstance(next(iter(batch_data.values())), list)
        ):
            return self._deserialise_batch(batch_data)
        return batch_data

    def _deserialise_batch(self, batch_data: dict) -> dict:
        """Convert a serialised batch back to device tensors."""
        import torch

        device = next(self._trainer.actor.model.parameters()).device
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

    # -- Stage 2: forward passes (critic + log-probs) ---------------------

    def _compute_logps(self, ctx: _StepContext, rollout_batch: dict) -> None:
        """Compute critic values and all conditional log-probabilities.

        Mutates ``rollout_batch`` in-place with values, prox_logp,
        ref_logp, and teacher_logp as applicable.
        """
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
                trainer.critic.get_device_stats().log("critic values")

        if ctx.config.actor.should_compute_prox_logp():
            with (
                stats_tracker.record_timing("recompute_logp"),
                perf_tracer.trace_scope(
                    "train.recompute_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                rollout_batch["prox_logp"] = trainer.actor.compute_logp(rollout_batch)
                trainer.actor.get_device_stats().log("recompute logp")

        if trainer.ref is not None:
            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp", category=Category.COMPUTE, args=sa
                ),
            ):
                rollout_batch["ref_logp"] = trainer.ref.compute_logp(rollout_batch)
                trainer.ref.get_device_stats().log("ref logp")

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
                trainer.teacher.get_device_stats().log("teacher logp")

    # -- Stage 3: advantage estimation ------------------------------------

    def _compute_advantages(self, ctx: _StepContext, rollout_batch: dict) -> dict:
        """Compute advantages from the rollout batch.  Returns adv_batch."""
        trainer = ctx.trainer
        with (
            stats_tracker.record_timing("compute_advantage"),
            perf_tracer.trace_scope(
                "train.compute_advantage", category=Category.COMPUTE, args=ctx.step_args
            ),
        ):
            adv_batch = trainer.actor.compute_advantages(rollout_batch)
            trainer.actor.get_device_stats().log("compute advantages")
        return adv_batch

    # -- Stage 5a: actor PPO update ---------------------------------------

    def _ppo_update(self, ctx: _StepContext, adv_batch: dict) -> None:
        """Actor PPO update and learning-rate scheduler step."""
        trainer = ctx.trainer
        with (
            stats_tracker.record_timing("train_step"),
            perf_tracer.trace_scope(
                "train.ppo_update", category=Category.COMPUTE, args=ctx.step_args
            ),
        ):
            trainer.actor.ppo_update(adv_batch)
            trainer.actor.step_lr_scheduler()
            trainer.actor.get_device_stats().log("ppo update")

    # -- Stage 5b: critic update (conditional) ----------------------------

    def _critic_update(self, ctx: _StepContext, adv_batch: dict) -> None:
        """Critic PPO update and learning-rate scheduler step (if present)."""
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
            trainer.critic.get_device_stats().log("ppo critic update")

    # -- Stage 6: weight sync to inference engines -------------------------

    def _sync_weights(self, ctx: _StepContext) -> None:
        """Pause rollout, update inference weights, bump version, resume."""
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

    # -- Stage 7: save, evaluate, clear -----------------------------------

    def _save_and_evaluate(
        self,
        ctx: _StepContext,
        rollout_batch: dict,
        adv_batch: dict,
        eval_workflow,
        eval_workflow_kwargs,
    ) -> None:
        """Save checkpoints, run evaluation, clear batches."""
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

    # -- Stage 8: log stats and resume rollout ----------------------------

    def _log_and_resume(self, ctx: _StepContext) -> None:
        """Export stats, resume rollout, save perf tracer."""
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

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    @endpoint
    def shutdown(self) -> None:
        if self._trainer is not None:
            logger.info(f"TrainerActor[rank={self._rank}] shutting down")
            try:
                self._trainer.close()
            except Exception as e:
                logger.warning(f"Error during trainer close: {e}")
            self._trainer = None
