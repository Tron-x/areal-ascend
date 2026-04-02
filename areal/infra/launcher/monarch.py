"""
Monarch-based launcher for AReaL — Phase 2: Step-Level Training Control.

Uses Monarch actors to manage the full training lifecycle:
  - VLLMServerActor: manages vLLM inference server as a subprocess
  - TrainerActor: runs FSDP training in-process with step-level control

The Monarch orchestrator drives each training step, giving full visibility
and control over the GRPO/PPO loop.

Usage:
    python -m areal.infra.launcher.monarch \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml

Architecture:
    MonarchOrchestrator (main process, no NPU)
      ├── InferenceProcMesh (1 proc, NPU hidden from parent)
      │     └── VLLMServerActor → subprocess: vLLM server (owns NPU 0)
      └── TrainingProcMesh (1 proc per training NPU)
            └── TrainerActor — in-process FSDPEngine (owns NPU 1)
"""

import asyncio
import importlib.util
import os
import signal
import subprocess
import sys

import psutil

from areal.api import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    InferenceEngineConfig,
    RecoverConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.platforms import current_platform
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    get_scheduling_spec,
    get_thread_env_vars,
    validate_config_for_launcher,
    wait_llm_server_addrs,
)
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports
from areal.utils.recover import check_if_recover

logger = logging.getLogger("MonarchLauncher")


# ---------------------------------------------------------------------------
# Bootstrap functions (run inside each Monarch-spawned process)
# ---------------------------------------------------------------------------


def npu_bootstrap_no_device():
    """Bootstrap for VLLMServerActor: hide NPUs from the Monarch process.

    The vLLM subprocess manages its own NPU devices. Setting
    ASCEND_RT_VISIBLE_DEVICES to empty prevents the parent Monarch process
    from initializing an NPU context (which would consume device memory
    and leave 0 bytes for the child vLLM workers).
    """

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""

    return _bootstrap


def npu_training_bootstrap(device_id: int):
    """Bootstrap for TrainerActor: bind to a specific NPU and initialize it.

    Training runs in-process (not via subprocess), so the actor needs
    direct NPU access for model weights and forward/backward computation.

    Also forces ``fork`` multiprocessing so that AReaL's DataLoader workers
    (which may hold non-picklable closures) can start without errors.
    """

    def _bootstrap():
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _kill_proc_tree(pid: int, sig=signal.SIGTERM, timeout: int = 15):
    """Send signal to a process tree, escalate to SIGKILL on timeout."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(sig)
        parent.send_signal(sig)
        _, alive = psutil.wait_procs([parent] + children, timeout=timeout)
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass


def _resolve_device_ids(n_total: int, env_var: str) -> list:
    """Parse available device IDs from environment or default to 0..n-1."""
    raw = os.environ.get(env_var, "")
    if raw:
        return [d.strip() for d in raw.split(",") if d.strip()]
    return [str(i) for i in range(n_total)]


# ---------------------------------------------------------------------------
# Monarch Actors
# ---------------------------------------------------------------------------

from monarch.actor import Actor, endpoint, this_host  # noqa: E402


class VLLMServerActor(Actor):
    """Manages a vLLM inference server subprocess.

    Unchanged from Phase 1 — vLLM runs as a subprocess because it is a
    full async HTTP server with its own worker processes.
    """

    def __init__(self, server_cmd: str, env_vars: dict, log_path: str):
        self._server_cmd = server_cmd
        self._env_vars = env_vars
        self._log_path = log_path
        self._process = None

    @endpoint
    def start(self) -> int:
        """Start vLLM server subprocess. Returns PID."""
        env = os.environ.copy()
        env.update(self._env_vars)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        cmd = (
            f"set -o pipefail; stdbuf -oL {self._server_cmd} "
            f"2>&1 | tee -a {self._log_path}"
        )
        logger.info(f"VLLMServerActor starting: {self._server_cmd}")
        self._process = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
        )
        logger.info(f"VLLMServerActor subprocess started, PID={self._process.pid}")
        return self._process.pid

    @endpoint
    def health(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @endpoint
    def shutdown(self) -> None:
        if self._process is not None and self._process.poll() is None:
            logger.info(f"VLLMServerActor stopping PID {self._process.pid}")
            _kill_proc_tree(self._process.pid)


class TrainerActor(Actor):
    """In-process FSDP training actor with step-level control.

    Phase 2 architecture: the model lives inside this actor process.
    Forward/backward passes, optimizer steps, and weight sync all happen
    in-process.  The Monarch orchestrator drives each training step via
    the ``train_step`` endpoint.

    Lifecycle:
        __init__  → store parameters
        initialize → torch.distributed init, model load, FSDP wrap
        train_step → one full GRPO iteration
        shutdown   → release resources
    """

    def __init__(
        self,
        cli_args: list,
        env_vars: dict,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        self._cli_args = cli_args
        self._env_vars = env_vars
        self._rank = rank
        self._world_size = world_size
        self._master_addr = master_addr
        self._master_port = master_port
        self._trainer = None
        self._train_kwargs: dict = {}
        self._max_steps = 0
        self._steps_per_epoch = 1

    @endpoint
    def initialize(self) -> dict:
        """Set up distributed env, load model, create PPOTrainer.

        Dynamically imports the experiment script (e.g. gsm8k_rl.py) and
        runs its ``main()`` with a monkeypatched ``PPOTrainer.train()`` so
        we capture the fully-initialized trainer without starting the loop.
        """
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ.update(self._env_vars)

        logger.info(
            f"TrainerActor[rank={self._rank}] initializing: "
            f"WORLD_SIZE={self._world_size}, "
            f"ASCEND_RT_VISIBLE_DEVICES="
            f"{os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'N/A')}"
        )

        from areal import PPOTrainer

        _captured: dict = {}
        _orig_train = PPOTrainer.train
        _orig_exit = PPOTrainer.__exit__

        def _intercept_train(trainer_self, **kwargs):
            _captured["trainer"] = trainer_self
            _captured["kwargs"] = kwargs

        def _intercept_exit(trainer_self, exc_type, exc_value, traceback):
            pass

        PPOTrainer.train = _intercept_train
        PPOTrainer.__exit__ = _intercept_exit

        try:
            script_path = self._cli_args[0]
            remaining_args = self._cli_args[1:]

            spec = importlib.util.spec_from_file_location(
                "_monarch_experiment", script_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main(remaining_args)
        finally:
            PPOTrainer.train = _orig_train
            PPOTrainer.__exit__ = _orig_exit

        self._trainer = _captured["trainer"]
        self._train_kwargs = _captured["kwargs"]

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
            f"max_steps={self._max_steps}, start_step={start_step}, "
            f"steps_per_epoch={self._steps_per_epoch}"
        )

        return {
            "status": "ready",
            "rank": self._rank,
            "max_steps": self._max_steps,
            "start_step": start_step,
            "steps_per_epoch": self._steps_per_epoch,
        }

    @endpoint
    def train_step(self, global_step: int) -> dict:
        """Run one GRPO/PPO training iteration.

        Replicates the loop body of ``PPOTrainer.train()``
        (rl_trainer.py lines 336-543).
        """
        trainer = self._trainer
        config = trainer.config

        epoch = global_step // self._steps_per_epoch
        step_in_epoch = global_step % self._steps_per_epoch

        workflow = self._train_kwargs.get("workflow")
        workflow_kwargs = self._train_kwargs.get("workflow_kwargs")
        eval_workflow = self._train_kwargs.get("eval_workflow")
        eval_workflow_kwargs = self._train_kwargs.get("eval_workflow_kwargs")
        dynamic_filter_fn = self._train_kwargs.get("dynamic_filter_fn")

        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        step_args = {"global_step": global_step, "epoch_step": step_in_epoch}

        # --- 1. Rollout (generate via vLLM HTTP) ---
        with (
            stats_tracker.record_timing("rollout"),
            perf_tracer.trace_scope(
                "train.rollout", category=Category.COMPUTE, args=step_args
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

        # --- 2. Critic values (optional) ---
        if trainer.critic is not None:
            with (
                stats_tracker.record_timing("critic_values"),
                perf_tracer.trace_scope(
                    "train.compute_values",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["values"] = trainer.critic.compute_values(rollout_batch)
                trainer.critic.get_device_stats().log("critic values")

        # --- 3. Recompute proximal log-probs ---
        if config.actor.should_compute_prox_logp():
            with (
                stats_tracker.record_timing("recompute_logp"),
                perf_tracer.trace_scope(
                    "train.recompute_logp",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["prox_logp"] = trainer.actor.compute_logp(rollout_batch)
                trainer.actor.get_device_stats().log("recompute logp")

        # --- 4. Reference log-probs (optional) ---
        if trainer.ref is not None:
            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["ref_logp"] = trainer.ref.compute_logp(rollout_batch)
                trainer.ref.get_device_stats().log("ref logp")

        # --- 5. Teacher log-probs (optional) ---
        if trainer.teacher is not None:
            with (
                stats_tracker.record_timing("teacher_logp"),
                perf_tracer.trace_scope(
                    "train.teacher_logp",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["teacher_logp"] = trainer.teacher.compute_logp(
                    rollout_batch
                )
                rollout_batch["rl_loss_weight"] = config.teacher.rl_loss_weight
                rollout_batch["distill_loss_weight"] = (
                    config.teacher.distill_loss_weight
                )
                trainer.teacher.get_device_stats().log("teacher logp")

        # --- 6. Compute advantages (GRPO) ---
        with (
            stats_tracker.record_timing("compute_advantage"),
            perf_tracer.trace_scope(
                "train.compute_advantage",
                category=Category.COMPUTE,
                args=step_args,
            ),
        ):
            adv_batch = trainer.actor.compute_advantages(rollout_batch)
            trainer.actor.get_device_stats().log("compute advantages")

        # --- 7. Wait for async checkpoint staging ---
        trainer.saver.maybe_wait_for_staging()

        # --- 8. PPO/GRPO policy update ---
        with (
            stats_tracker.record_timing("train_step"),
            perf_tracer.trace_scope(
                "train.ppo_update",
                category=Category.COMPUTE,
                args=step_args,
            ),
        ):
            trainer.actor.ppo_update(adv_batch)
            trainer.actor.step_lr_scheduler()
            trainer.actor.get_device_stats().log("ppo update")

        # --- 9. Critic update (optional) ---
        if trainer.critic is not None:
            with (
                stats_tracker.record_timing("critic_train_step"),
                perf_tracer.trace_scope(
                    "train.critic_ppo_update",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                trainer.critic.ppo_update(adv_batch)
                trainer.critic.step_lr_scheduler()
                trainer.critic.get_device_stats().log("ppo critic update")

        # --- 10. Pause inference for weight update ---
        trainer.rollout.pause()

        # --- 11. Sync weights to inference (xccl broadcast) ---
        with (
            stats_tracker.record_timing("update_weights"),
            perf_tracer.trace_scope(
                "train.update_weights",
                category=Category.COMM,
                args=step_args,
            ),
        ):
            new_version = global_step + 1
            versioned_meta = trainer.weight_update_meta.with_version(new_version)
            trainer.actor.update_weights(versioned_meta)
            trainer.actor.set_version(new_version)
            if trainer.critic is not None:
                trainer.critic.set_version(new_version)
            trainer.rollout.set_version(new_version)
            if trainer.eval_rollout is not None:
                trainer.eval_rollout.set_version(new_version)

        # --- 12. Save HF checkpoint ---
        with (
            stats_tracker.record_timing("save"),
            perf_tracer.trace_scope("train.save", category=Category.IO, args=step_args),
        ):
            trainer._save_hf(
                epoch=epoch, epoch_step=step_in_epoch, global_step=global_step
            )

        # --- 13. Save recovery checkpoint ---
        with (
            stats_tracker.record_timing("checkpoint_for_recover"),
            perf_tracer.trace_scope(
                "train.checkpoint", category=Category.IO, args=step_args
            ),
        ):
            trainer._save_recover_checkpoint(
                epoch=epoch, epoch_step=step_in_epoch, global_step=global_step
            )

        # --- 14. Evaluation ---
        with (
            stats_tracker.record_timing("eval"),
            perf_tracer.trace_scope(
                "train.eval", category=Category.COMPUTE, args=step_args
            ),
        ):
            trainer._evaluate(
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                epoch=epoch,
                epoch_step=step_in_epoch,
                global_step=global_step,
            )

        # --- 15. Clear batch tensors ---
        with (
            stats_tracker.record_timing("clear_batches"),
            perf_tracer.trace_scope(
                "train.clear_batches",
                category=Category.INSTR,
                args=step_args,
            ),
        ):
            trainer.actor.clear_batches(rollout_batch, adv_batch)

        # --- 16. Export and commit stats ---
        with perf_tracer.trace_scope(
            "train.log_stats", category=Category.INSTR, args=step_args
        ):
            trainer._export_and_commit_stats(
                epoch=epoch, epoch_step=step_in_epoch, global_step=global_step
            )

        # --- 17. Resume rollout ---
        trainer.rollout.resume()

        # --- 18. Save perf tracer ---
        trainer._save_perf_tracer(step=global_step)

        return {
            "global_step": global_step,
            "epoch": epoch,
            "epoch_step": step_in_epoch,
        }

    @endpoint
    def shutdown(self) -> None:
        """Release all training resources."""
        if self._trainer is not None:
            logger.info(f"TrainerActor[rank={self._rank}] shutting down")
            try:
                self._trainer.close()
            except Exception as e:
                logger.warning(f"Error during trainer close: {e}")
            self._trainer = None


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------


async def monarch_main_async(config, run_id: int = 0):
    """Core async orchestration — step-level training control."""
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_launcher(config)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )
    alloc_mode = AllocationMode.from_str(config.allocation_mode)

    logger.info(
        f"MonarchLauncher: experiment_name={config.experiment_name}, "
        f"trial_name={config.trial_name}, fileroot={config.cluster.fileroot}, "
        f"run_id={run_id}, is_recover_run={is_recover_run}"
    )

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    env_var = current_platform.device_control_env_var
    all_device_ids = _resolve_device_ids(config.cluster.n_gpus_per_node, env_var)

    gen_gpu_count = (
        alloc_mode.gen.pp_size * alloc_mode.gen.tp_size * alloc_mode.gen.dp_size
    )
    train_gpu_count = (
        alloc_mode.train.world_size
        if alloc_mode.type_ != AllocationType.LLM_SERVER_ONLY
        else 0
    )

    inf_device_ids = all_device_ids[:gen_gpu_count]
    train_device_ids = all_device_ids[gen_gpu_count : gen_gpu_count + train_gpu_count]

    logger.info(
        f"Device allocation: inference={inf_device_ids}, training={train_device_ids}"
    )

    fileroot = config.cluster.fileroot
    user = os.environ.get("USER", "root")
    log_dir = f"{fileroot}/logs/{user}/{config.experiment_name}/{config.trial_name}"
    os.makedirs(log_dir, exist_ok=True)

    host = this_host()
    vllm_actor = None
    trainer_actor = None

    try:
        # ================================================================
        # Phase A: Launch vLLM inference server (subprocess, unchanged)
        # ================================================================
        server_addrs = []
        if alloc_mode.gen_backend in ("sglang", "vllm"):
            if alloc_mode.gen_backend == "vllm":
                config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
                random_seed = config.vllm.seed
            else:
                config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
                random_seed = config.sglang.random_seed

            config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)

            backend_spec = {
                "sglang": {
                    "module": "areal.infra.launcher.sglang_server",
                    "seed_arg": "sglang.random_seed",
                },
                "vllm": {
                    "module": "areal.infra.launcher.vllm_server",
                    "seed_arg": "vllm.seed",
                },
            }
            spec = backend_spec[alloc_mode.gen_backend]
            server_cmd = (
                f"python3 -m {spec['module']} "
                f"{' '.join(sys.argv[1:])} {spec['seed_arg']}={random_seed}"
            )

            rollout_spec = get_scheduling_spec(config.rollout)
            rollout_env_vars = rollout_spec.env_vars
            thread_env = get_thread_env_vars(
                cpus_per_task=rollout_spec.cpu,
                existing_env_vars=rollout_env_vars,
            )
            vllm_env = {
                **BASE_ENVIRONS,
                **thread_env,
                **rollout_env_vars,
                env_var: ",".join(inf_device_ids),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
                "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            }

            logger.info(f"Spawning inference ProcMesh on NPU {inf_device_ids}")
            inf_procs = host.spawn_procs(
                per_host={"npu": 1},
                bootstrap=npu_bootstrap_no_device(),
                name="inference",
            )

            vllm_actor = inf_procs.spawn(
                "vllm_server",
                VLLMServerActor,
                server_cmd=server_cmd,
                env_vars=vllm_env,
                log_path=os.path.join(log_dir, "llm_server.log"),
            )

            logger.info("Starting vLLM server via Monarch actor ...")
            pid = await vllm_actor.start.call_one()
            logger.info(f"vLLM server subprocess started (PID={pid})")

            try:
                server_addrs = wait_llm_server_addrs(
                    config.experiment_name,
                    config.trial_name,
                    n_rollout_servers=alloc_mode.gen.dp_size,
                )
            except (TimeoutError, KeyboardInterrupt) as e:
                await vllm_actor.shutdown.call_one()
                raise e

            logger.info(
                f"vLLM servers ready: AREAL_LLM_SERVER_ADDRS={','.join(server_addrs)}"
            )

        # ================================================================
        # Phase B: Spawn in-process trainer on training NPU
        # ================================================================
        if alloc_mode.type_ != AllocationType.LLM_SERVER_ONLY:
            nprocs = alloc_mode.train.world_size
            master_port = find_free_ports(1, (10000, 50000))[0]

            actor_spec = get_scheduling_spec(config.actor)
            actor_env_vars = actor_spec.env_vars
            thread_env = get_thread_env_vars(
                cpus_per_task=actor_spec.cpu,
                existing_env_vars=actor_env_vars,
            )

            tms_env_vars = {}
            if config.get("enable_offload", False):
                from areal.utils.offload import get_tms_env_vars

                tms_env_vars = get_tms_env_vars()

            trainer_env = {
                **BASE_ENVIRONS,
                **thread_env,
                **actor_env_vars,
                **tms_env_vars,
                "AREAL_LLM_SERVER_ADDRS": ",".join(server_addrs),
                "AREAL_SPMD_MODE": "1",
                env_var: ",".join(train_device_ids),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
                "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            }

            if alloc_mode.gen_backend == "sglang":
                trainer_env["NCCL_CUMEM_ENABLE"] = "0"
                trainer_env["NCCL_NVLS_ENABLE"] = "0"

            train_device_id = int(train_device_ids[0])
            logger.info(
                f"Spawning training ProcMesh on NPU {train_device_id} (rank 0/{nprocs})"
            )
            train_procs = host.spawn_procs(
                per_host={"npu": 1},
                bootstrap=npu_training_bootstrap(train_device_id),
                name="training",
            )

            trainer_actor = train_procs.spawn(
                "trainer",
                TrainerActor,
                cli_args=sys.argv[1:],
                env_vars=trainer_env,
                rank=0,
                world_size=nprocs,
                master_addr="localhost",
                master_port=master_port,
            )

            # ============================================================
            # Phase C: Initialize trainer (model load, FSDP, optimizer)
            # ============================================================
            logger.info("Initializing TrainerActor (loading model, FSDP setup) ...")
            info = await trainer_actor.initialize.call_one()
            max_steps = info["max_steps"]
            start_step = info["start_step"]
            logger.info(
                f"TrainerActor ready: max_steps={max_steps}, start_step={start_step}"
            )

            # ============================================================
            # Phase D: Step-by-step training loop
            # ============================================================
            logger.info(f"Starting training loop: steps {start_step} → {max_steps}")
            for global_step in range(start_step, max_steps):
                result = await trainer_actor.train_step.call_one(global_step)
                logger.info(
                    f"[Step {global_step + 1}/{max_steps}] "
                    f"epoch={result['epoch']}, "
                    f"epoch_step={result['epoch_step']}"
                )

            logger.info("Training completed successfully.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down ...")
    except Exception as e:
        logger.error(f"MonarchLauncher error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up Monarch actors ...")
        if trainer_actor is not None:
            try:
                await trainer_actor.shutdown.call_one()
            except Exception:
                pass
        if vllm_actor is not None:
            try:
                await vllm_actor.shutdown.call_one()
            except Exception:
                pass
        logger.info("MonarchLauncher shutdown complete.")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def monarch_main(config, run_id: int = 0):
    """Synchronous wrapper around the async Monarch launcher."""
    from monarch._src.actor.actor_mesh import context

    context()
    asyncio.run(monarch_main_async(config, run_id))


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    monarch_main(config, run_id=0)


if __name__ == "__main__":
    main()
