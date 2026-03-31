"""
Monarch-based launcher for AReaL -- topology-aware distributed RL.

Uses Monarch actors for the full training lifecycle:
  - GeneratorActor: embeds vLLM AsyncLLM with AReaLMonarchExecutor
  - RewardActor: dedicated reward computation on CPU ProcMesh
  - SandboxActor: isolated Python code execution on CPU ProcMesh
  - AgentActor: multi-turn agent orchestrating Generator+Sandbox+Reward
  - ReplayBufferActor: async experience replay for pipeline parallelism
  - RolloutActor: independent rollout production (CPU ProcMesh, no model)
  - TrainerActor: runs FSDP training in-process with step-level control
  - All communication via Monarch RPC (no HTTP)

Topology-aware placement (via topology.py):
  The launcher supports arbitrary device configurations through
  ClusterTopology.  Device placement is computed from allocation_mode
  and cluster config, not hardcoded.

XCCL weight-update alloc_mode (via weight_sync.py):
  For ``type="xccl"``, ``WeightUpdateMeta.alloc_mode.gen`` must match how many
  vLLM workers join the weight-update group (``GeneratorActor`` collective RPC).
  By default we rebuild alloc_mode from the cluster ``allocation_mode`` inference
  parallel strategy plus training world size; optional env
  ``MONARCH_INFERENCE_XCCL_PARTICIPANTS`` forces a flat ``d{N}p1t1`` inference
  half. See ``weight_sync.resolve_xccl_alloc_mode``.

  Single-node: devices linearly partitioned (inference first, then training)
  Multi-node:  MONARCH_WORKERS env var or cluster.monarch_workers config
               provides worker addresses; symmetric or role-split placement.

Supported configurations:
  allocation_mode         n_gpus_per_node    Layout
  ──────────────────────  ─────────────────  ──────────────────────────
  vllm:d1p1t1+d1p1t1     2                  1 inf + 1 train
  vllm:d2p1t1+d2p1t1     4                  2 inf + 2 train
  vllm:d1p1t1+d3p1t1     4                  1 inf + 3 train
  vllm:d4p1t1+d4p1t1     8                  4 inf + 4 train
  vllm:d2p1t1+d6p1t1     8                  2 inf + 6 train
  vllm:d1p1t4+d4p1t1     8                  1 inf (TP=4) + 4 train

Multi-node (requires MONARCH_WORKERS):
  allocation_mode         n_nodes  Layout
  ──────────────────────  ───────  ──────────────────────────
  vllm:d4p1t1+d4p1t1     2        each node: 4 inf + 4 train
  vllm:d8p1t1+d8p1t1     2        each node: 8 inf + 8 train

Usage:
    # 4+4 (8 NPU, single node):
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml

    # 1+1 (2 NPU):
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        "allocation_mode=vllm:d1p1t1+d1p1t1" \\
        "cluster.n_gpus_per_node=2"

    # 2+6 (8 NPU):
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        "allocation_mode=vllm:d2p1t1+d6p1t1"

    # Multi-node (2 nodes × 8 NPU):
    MONARCH_WORKERS=tcp://node0:29600,tcp://node1:29600 \\
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        "cluster.n_nodes=2"

Architecture (4+4 single-node example):
    MonarchOrchestrator (main process, no NPU)
      ├── GeneratorProcMesh (1 proc, NPUs 0-3 visible)
      │     ├── WorkerRegistry (actor)
      │     └── GeneratorActor → AsyncLLM → EngineCore subprocess
      │           └── AReaLMonarchExecutor → vLLM Worker ProcMesh (NPU 0-3)
      ├── RewardProcMesh (1 proc, CPU only)
      │     └── RewardActor → reward_fn() via Monarch RPC
      ├── SandboxProcMesh (1 proc, CPU only)
      │     └── SandboxActor → subprocess code execution
      ├── AgentProcMesh (1 proc, CPU only)
      │     └── AgentActor → multi-turn orchestration
      ├── ReplayBufferProcMesh (1 proc, CPU only)
      │     └── ReplayBufferActor → async experience replay
      ├── RolloutProcMesh (1 proc, CPU only)
      │     └── RolloutActor → dataloader + WorkflowExecutor + MonarchVLLMEngine
      │           ├── do_rollout() → produces batches → ReplayBuffer
      │           └── (no model loaded, inference via GeneratorActor RPC)
      └── TrainingProcMesh (nprocs processes, NPUs 4-7)
            ├── TrainerActor[rank=0] ─┐
            ├── TrainerActor[rank=1]  ├── FSDP via HCCL all-reduce
            ├── TrainerActor[rank=2]  │
            └── TrainerActor[rank=3] ─┘
                  └── train_on_batch() ← consumes batches ← ReplayBuffer
"""

import asyncio
import importlib.util
import os
import signal
import sys

import psutil

from areal.api import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    InferenceEngineConfig,
    RecoverConfig,
    SGLangConfig,
    conf_as_dict,
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
)
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports
from areal.utils.recover import check_if_recover

logger = logging.getLogger("MonarchPlugin")


def _ensure_ascend_custom_opp_path():
    """Pre-set ASCEND_CUSTOM_OPP_PATH so Monarch-spawned processes inherit it.

    vllm_ascend ships custom CANN ops (e.g. aclnnAddRmsNormBias) under
    its _cann_ops_custom directory.  The CANN runtime reads
    ASCEND_CUSTOM_OPP_PATH to locate them.  Normally vllm_ascend sets
    this in its platform init, but Monarch worker processes are forked
    from the host agent (main process) which never imports vllm_ascend.
    Setting it here ensures all child processes inherit the value.
    """
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        return
    try:
        import vllm_ascend
        pkg_dir = os.path.dirname(os.path.realpath(vllm_ascend.__file__))
        custom_opp = os.path.join(
            pkg_dir, "_cann_ops_custom", "vendors", "vllm-ascend"
        )
        if os.path.isdir(custom_opp):
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp
            logger.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp}")
    except ImportError:
        pass


_ensure_ascend_custom_opp_path()


# ---------------------------------------------------------------------------
# Bootstrap functions
# ---------------------------------------------------------------------------

def generator_bootstrap(device_ids_str: str):
    """Bootstrap for GeneratorActor: make NPU devices visible but do NOT
    initialise an NPU context.  AsyncLLM only needs to validate the config
    (e.g. ``torch.npu.device_count()``); actual compute happens on the
    workers spawned by AReaLMonarchExecutor.

    Also propagates LD_LIBRARY_PATH to ensure CANN/ATB libraries are
    available in the EngineCore subprocess and worker processes.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_ids_str
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
    return _bootstrap


def npu_training_bootstrap(device_id: int):
    """Bootstrap for TrainerActor (single-process): bind to a specific NPU.
    Also forces ``fork`` multiprocessing for AReaL DataLoader compatibility.
    Propagates LD_LIBRARY_PATH for CANN/ATB library availability.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    def _bootstrap():
        import multiprocessing as _mp
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    return _bootstrap


def npu_training_bootstrap_multi(all_device_ids: str):
    """Bootstrap for multi-process FSDP training: all training NPUs visible.

    Each spawned process sees ALL training devices.  The actual device
    selection (``torch.npu.set_device(local_rank)``) happens inside
    ``TrainerActor.initialize()`` after the Monarch rank is known.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    def _bootstrap():
        import multiprocessing as _mp
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        if ld_path:
            os.environ["LD_LIBRARY_PATH"] = ld_path
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = all_device_ids
        import torch
        import torch_npu  # noqa: F401
    return _bootstrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kill_proc_tree(pid: int, sig=signal.SIGTERM, timeout: int = 15):
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


def _build_vllm_cli_args(config, alloc_mode) -> list[str]:
    """Build the CLI arg list that would normally be passed to
    ``areal.engine.vllm_ext.areal_vllm_server``.

    We reuse ``vLLMConfig.build_args`` and convert to a flat string list
    so that GeneratorActor can parse them with vLLM's own argument parser.
    """
    args_dict = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=alloc_mode.gen.tp_size,
        pp_size=alloc_mode.gen.pp_size,
    )
    cli: list[str] = []
    for k, v in args_dict.items():
        if v is None or v is False or v == "" or (isinstance(v, list) and not v):
            continue
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cli.append(flag)
        elif isinstance(v, list):
            cli.append(flag)
            cli.extend(str(x) for x in v)
        else:
            cli.append(flag)
            cli.append(str(v))
    return cli


# ---------------------------------------------------------------------------
# Monarch Actors
# ---------------------------------------------------------------------------

from monarch.actor import Actor, endpoint  # noqa: E402

from areal.monarch_plugin.agent_actor import AgentActor  # noqa: E402
from areal.monarch_plugin.executor import WorkerRegistry  # noqa: E402
from areal.monarch_plugin.generator_actor import GeneratorActor  # noqa: E402
from areal.monarch_plugin.replay_buffer_actor import ReplayBufferActor  # noqa: E402
from areal.monarch_plugin.reward_actor import RewardActor  # noqa: E402
from areal.monarch_plugin.rollout_actor import RolloutActor  # noqa: E402
from areal.monarch_plugin.sandbox_actor import SandboxActor  # noqa: E402
from areal.monarch_plugin.weight_sync import (  # noqa: E402
    resolve_xccl_alloc_mode,
)


class TrainerActor(Actor):
    """In-process FSDP training actor with Monarch RPC rollout and reward.

    Phase 5 enhancement:
      - Accepts ``generator_actor``, ``reward_actor``, and ``agent_actor``
      - Monkey-patches ``PPOTrainer._init_rollout`` to create
        ``MonarchVLLMEngine`` with all actor references
      - ``xccl_weight_update_alloc_mode``: ``AllocationMode`` used when patching
        FSDP XCCL connect (see
        ``weight_sync.resolve_xccl_alloc_mode``)
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
                f"TrainerActor auto-detected rank={self._rank} "
                f"from ProcMesh coordinate"
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

        def _patched_init_rollout(trainer_self, rollout_config, is_eval=False, lora_path=None):
            """Replace RemotevLLMEngine with MonarchVLLMEngine."""
            from areal.utils.environ import is_single_controller
            if is_single_controller():
                return _orig_init_rollout(trainer_self, rollout_config, is_eval, lora_path)

            config = rollout_config
            engine = MonarchVLLMEngine(
                config, generator_ref,
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
            start_step = (
                self._trainer.recover_info.last_step_info.next().global_step
            )

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
        """Run one GRPO/PPO training iteration (rollout + train combined).

        Kept for backward compatibility.  Phase 6 uses ``do_rollout`` +
        ``train_on_batch`` instead.
        """
        batch_data = self._do_rollout_impl(global_step)
        return self._train_on_batch_impl(batch_data, global_step)

    # ------------------------------------------------------------------
    # Phase 6: split endpoints for async rollout / training pipeline
    # ------------------------------------------------------------------

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        """Run rollout only, serialise the batch for ReplayBuffer.

        This endpoint uses the Generator NPU (via Monarch RPC) but does
        NOT touch the training NPU, so it can overlap with a prior
        ``train_on_batch`` call.

        Returns
        -------
        dict
            Serialised rollout batch (numpy arrays converted to lists for
            Monarch RPC transport).
        """
        return self._do_rollout_impl(global_step)

    @endpoint
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        """Train on a pre-produced rollout batch from the ReplayBuffer.

        Parameters
        ----------
        batch_data : dict
            Serialised rollout batch from ``do_rollout`` / ReplayBuffer.
        global_step : int
            Current training step.
        """
        return self._train_on_batch_impl(batch_data, global_step)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _do_rollout_impl(self, global_step: int) -> dict:
        """Produce a rollout batch and serialise it."""
        import numpy as np
        import torch

        trainer = self._trainer
        config = trainer.config
        step_in_epoch = global_step % self._steps_per_epoch

        workflow = self._train_kwargs.get("workflow")
        workflow_kwargs = self._train_kwargs.get("workflow_kwargs")
        dynamic_filter_fn = self._train_kwargs.get("dynamic_filter_fn")

        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        step_args = {"global_step": global_step, "epoch_step": step_in_epoch}

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

        serialised = {}
        for k, v in rollout_batch.items():
            if isinstance(v, torch.Tensor):
                serialised[k] = v.cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                serialised[k] = v.tolist()
            else:
                serialised[k] = v
        return serialised

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

    def _train_on_batch_impl(self, batch_data: dict, global_step: int) -> dict:
        """Run training on a (possibly deserialised) rollout batch."""
        import torch

        trainer = self._trainer
        config = trainer.config
        epoch = global_step // self._steps_per_epoch
        step_in_epoch = global_step % self._steps_per_epoch

        eval_workflow = self._train_kwargs.get("eval_workflow")
        eval_workflow_kwargs = self._train_kwargs.get("eval_workflow_kwargs")

        from areal.utils import perf_tracer, stats_tracker
        from areal.utils.perf_tracer import Category

        step_args = {"global_step": global_step, "epoch_step": step_in_epoch}

        if isinstance(batch_data, dict) and batch_data and isinstance(
            next(iter(batch_data.values())), list
        ):
            rollout_batch = self._deserialise_batch(batch_data)
        else:
            rollout_batch = batch_data

        if trainer.critic is not None:
            with (
                stats_tracker.record_timing("critic_values"),
                perf_tracer.trace_scope(
                    "train.compute_values",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["values"] = trainer.critic.compute_values(
                    rollout_batch
                )
                trainer.critic.get_device_stats().log("critic values")

        if config.actor.should_compute_prox_logp():
            with (
                stats_tracker.record_timing("recompute_logp"),
                perf_tracer.trace_scope(
                    "train.recompute_logp",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["prox_logp"] = trainer.actor.compute_logp(
                    rollout_batch
                )
                trainer.actor.get_device_stats().log("recompute logp")

        if trainer.ref is not None:
            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp",
                    category=Category.COMPUTE,
                    args=step_args,
                ),
            ):
                rollout_batch["ref_logp"] = trainer.ref.compute_logp(
                    rollout_batch
                )
                trainer.ref.get_device_stats().log("ref logp")

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
                rollout_batch[
                    "distill_loss_weight"
                ] = config.teacher.distill_loss_weight
                trainer.teacher.get_device_stats().log("teacher logp")

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

        trainer.saver.maybe_wait_for_staging()

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

        trainer.rollout.pause()

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

        with (
            stats_tracker.record_timing("save"),
            perf_tracer.trace_scope(
                "train.save", category=Category.IO, args=step_args
            ),
        ):
            trainer._save_hf(
                epoch=epoch, epoch_step=step_in_epoch, global_step=global_step
            )

        with (
            stats_tracker.record_timing("checkpoint_for_recover"),
            perf_tracer.trace_scope(
                "train.checkpoint", category=Category.IO, args=step_args
            ),
        ):
            trainer._save_recover_checkpoint(
                epoch=epoch, epoch_step=step_in_epoch, global_step=global_step
            )

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

        with (
            stats_tracker.record_timing("clear_batches"),
            perf_tracer.trace_scope(
                "train.clear_batches",
                category=Category.INSTR,
                args=step_args,
            ),
        ):
            trainer.actor.clear_batches(rollout_batch, adv_batch)

        with perf_tracer.trace_scope(
            "train.log_stats", category=Category.INSTR, args=step_args
        ):
            trainer._export_and_commit_stats(
                epoch=epoch, epoch_step=step_in_epoch, global_step=global_step
            )

        trainer.rollout.resume()
        trainer._save_perf_tracer(step=global_step)

        return {
            "global_step": global_step,
            "epoch": epoch,
            "epoch_step": step_in_epoch,
        }

    @endpoint
    def shutdown(self) -> None:
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
    """Phase 6 orchestration: async rollout + ReplayBuffer + training pipeline."""
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
        f"MonarchPlugin Phase 6: experiment={config.experiment_name}, "
        f"trial={config.trial_name}, run_id={run_id}"
    )

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    env_var = current_platform.device_control_env_var

    from areal.monarch_plugin.topology import ClusterTopology
    topology = ClusterTopology.from_config(config, alloc_mode, env_var=env_var)
    placement = topology.placement

    inf_device_ids = placement.inference.all_device_ids
    train_device_ids = placement.training.all_device_ids

    logger.info(f"Topology:\n{topology.summary()}")

    fileroot = config.cluster.fileroot
    user = os.environ.get("USER", "root")
    log_dir = (
        f"{fileroot}/logs/{user}/{config.experiment_name}/{config.trial_name}"
    )
    os.makedirs(log_dir, exist_ok=True)

    host = topology.create_host_mesh()
    generator_actor = None
    reward_actor = None
    sandbox_actor = None
    agent_actor = None
    replay_buffer_actor = None
    rollout_actor = None
    trainer_actor = None
    gen_procs = None
    reward_procs = None
    sandbox_procs = None
    agent_procs = None
    replay_buffer_procs = None
    rollout_procs = None
    multi_rank = False

    try:
        # ================================================================
        # Phase A: Create GeneratorActor (embedded AsyncLLM, no HTTP)
        # ================================================================
        if alloc_mode.gen_backend == "vllm":
            config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
            config.rollout = to_structured_cfg(
                config.rollout, InferenceEngineConfig
            )

            vllm_cli_args = _build_vllm_cli_args(config, alloc_mode)
            logger.info(f"vLLM CLI args for GeneratorActor: {vllm_cli_args}")

            logger.info(
                f"Spawning GeneratorActor ProcMesh "
                f"(NPU visible: {inf_device_ids})"
            )
            gen_procs = host.spawn_procs(
                per_host={"npu": 1},
                bootstrap=generator_bootstrap(",".join(inf_device_ids)),
                name="generator",
            )

            worker_registry = gen_procs.spawn(
                "worker_registry", WorkerRegistry
            )

            generator_actor = gen_procs.spawn(
                "generator",
                GeneratorActor,
                vllm_cli_args=vllm_cli_args,
            )

            logger.info("Setting up GeneratorActor (AsyncLLM + MonarchExecutor) ...")
            await generator_actor.setup.call_one(
                host, worker_registry, inf_device_ids
            )
            logger.info("GeneratorActor ready (Monarch RPC mode)")
        else:
            raise ValueError(
                f"Monarch plugin only supports vllm backend, "
                f"got {alloc_mode.gen_backend}"
            )

        # ================================================================
        # Phase B: Create RewardActor (CPU ProcMesh, no NPU)
        # ================================================================
        logger.info("Spawning RewardActor ProcMesh (CPU only) ...")
        reward_procs = host.spawn_procs(
            per_host={"cpu": 1},
            name="reward",
        )
        reward_actor = reward_procs.spawn("reward", RewardActor)
        logger.info("RewardActor spawned (will lazy-load reward_fn on first call)")

        # ================================================================
        # Phase C: Create SandboxActor (CPU ProcMesh, code execution)
        # ================================================================
        logger.info("Spawning SandboxActor ProcMesh (CPU only) ...")
        sandbox_procs = host.spawn_procs(
            per_host={"cpu": 1},
            name="sandbox",
        )
        sandbox_actor = sandbox_procs.spawn("sandbox", SandboxActor)
        logger.info("SandboxActor spawned (subprocess-isolated code execution)")

        # ================================================================
        # Phase D: Create AgentActor (CPU ProcMesh, multi-turn orchestration)
        # ================================================================
        logger.info("Spawning AgentActor ProcMesh (CPU only) ...")
        agent_procs = host.spawn_procs(
            per_host={"cpu": 1},
            name="agent",
        )
        agent_actor = agent_procs.spawn(
            "agent",
            AgentActor,
            generator_actor=generator_actor,
            sandbox_actor=sandbox_actor,
            reward_actor=reward_actor,
        )
        logger.info(
            "AgentActor spawned (will lazy-setup on first episode from workflow)"
        )

        # ================================================================
        # Phase E: Create ReplayBufferActor (CPU ProcMesh, async replay)
        # ================================================================
        logger.info("Spawning ReplayBufferActor ProcMesh (CPU only) ...")
        replay_buffer_procs = host.spawn_procs(
            per_host={"cpu": 1},
            name="replay_buffer",
        )
        replay_buffer_actor = replay_buffer_procs.spawn(
            "replay_buffer", ReplayBufferActor, max_size=8,
        )
        logger.info("ReplayBufferActor spawned (max_size=8)")

        # ================================================================
        # Phase F: Create RolloutActor (CPU ProcMesh, independent rollout)
        # ================================================================
        logger.info("Spawning RolloutActor ProcMesh (CPU only) ...")
        rollout_procs = host.spawn_procs(
            per_host={"cpu": 1},
            name="rollout",
        )
        rollout_actor = rollout_procs.spawn(
            "rollout",
            RolloutActor,
            generator_actor=generator_actor,
            reward_actor=reward_actor,
            agent_actor=agent_actor,
        )
        logger.info("RolloutActor spawned (will setup on init)")

        # ================================================================
        # Phase G: Spawn TrainerActor with all actor references
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
                "AREAL_SPMD_MODE": "1",
                env_var: ",".join(train_device_ids),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                "TRANSFORMERS_OFFLINE": os.environ.get(
                    "TRANSFORMERS_OFFLINE", ""
                ),
                "VLLM_USE_MODELSCOPE": os.environ.get(
                    "VLLM_USE_MODELSCOPE", ""
                ),
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            }

            multi_rank = nprocs > 1
            xccl_alloc_mode = resolve_xccl_alloc_mode(
                config, alloc_mode, train_world_size=nprocs
            )

            if multi_rank:
                logger.info(
                    f"Spawning {nprocs} TrainerActors on NPUs "
                    f"{train_device_ids} (multi-process FSDP)"
                )
                train_procs = host.spawn_procs(
                    per_host={"npu": nprocs},
                    bootstrap=npu_training_bootstrap_multi(
                        ",".join(train_device_ids)
                    ),
                    name="training",
                )
            else:
                train_device_id = int(train_device_ids[0])
                logger.info(
                    f"Spawning TrainerActor on NPU {train_device_id} "
                    f"(single-process)"
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
                rank=-1 if multi_rank else 0,
                world_size=nprocs,
                master_addr=placement.master_addr,
                master_port=master_port,
                generator_actor=generator_actor,
                reward_actor=reward_actor,
                agent_actor=agent_actor,
                xccl_weight_update_alloc_mode=xccl_alloc_mode,
            )

            # ============================================================
            # Phase H: Initialise TrainerActor (model load, FSDP, engine)
            # ============================================================
            logger.info(
                "Initialising TrainerActor "
                f"({'multi-rank broadcast' if multi_rank else 'single'}: "
                f"model load + FSDP + MonarchVLLMEngine) ..."
            )
            if multi_rank:
                result_mesh = await trainer_actor.initialize.call()
                info = result_mesh.item(npu=0)
            else:
                info = await trainer_actor.initialize.call_one()
            max_steps = info["max_steps"]
            start_step = info["start_step"]
            logger.info(
                f"TrainerActor ready: max_steps={max_steps}, "
                f"start_step={start_step}"
            )

            # ============================================================
            # Phase I: Initialise RolloutActor (dataloader + engine, no model)
            # ============================================================
            logger.info("Initialising RolloutActor (dataset + engine) ...")
            rollout_info = await rollout_actor.setup.call_one(
                sys.argv[1:], train_dp_size=1,
            )
            logger.info(
                f"RolloutActor ready: "
                f"steps_per_epoch={rollout_info['steps_per_epoch']}"
            )

            # ============================================================
            # Phase J: Async rollout + training pipeline (true parallelism)
            #
            # RolloutActor (CPU ProcMesh) -> ReplayBuffer -> TrainerActor (NPU)
            # These are DIFFERENT actors on DIFFERENT ProcMeshes, so
            # do_rollout and train_on_batch run truly concurrently.
            # ============================================================
            logger.info(
                f"Starting async pipeline (true parallelism): "
                f"steps {start_step} -> {max_steps}"
            )

            rollout_step = start_step
            train_step_counter = start_step
            max_staleness = 3
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
                    await replay_buffer_actor.add_batch.call_one(
                        batch_data, step
                    )
                    buf_size = await replay_buffer_actor.buffer_size.call_one()
                    logger.info(
                        f"[Rollout] Step {step} batch added to buffer "
                        f"(buffer_size={buf_size})"
                    )
                    rollout_step += 1
                rollout_done.set()

            async def _training_loop():
                """Consume batches from ReplayBuffer and train on NPU."""
                nonlocal train_step_counter
                while train_step_counter < max_steps:
                    batch = await replay_buffer_actor.sample_batch.call_one(
                        train_step_counter, max_staleness
                    )
                    if batch is None:
                        if rollout_done.is_set():
                            logger.warning(
                                "[Training] Buffer empty and rollout done, "
                                "stopping training"
                            )
                            break
                        await asyncio.sleep(0.5)
                        continue

                    step = train_step_counter
                    if multi_rank:
                        result_mesh = await trainer_actor.train_on_batch.call(
                            batch, step
                        )
                        result = result_mesh.item(npu=0)
                    else:
                        result = await trainer_actor.train_on_batch.call_one(
                            batch, step
                        )
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

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down ...")
    except Exception as e:
        logger.error(f"MonarchPlugin error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up Monarch actors ...")
        if trainer_actor is not None:
            try:
                if multi_rank:
                    await trainer_actor.shutdown.call()
                else:
                    await trainer_actor.shutdown.call_one()
            except Exception:
                pass
        if agent_actor is not None:
            try:
                stats = await agent_actor.get_stats.call_one()
                logger.info(f"AgentActor stats: {stats}")
                await agent_actor.shutdown.call_one()
            except Exception:
                pass
        if sandbox_actor is not None:
            try:
                stats = await sandbox_actor.get_stats.call_one()
                logger.info(f"SandboxActor stats: {stats}")
                await sandbox_actor.shutdown.call_one()
            except Exception:
                pass
        if rollout_actor is not None:
            try:
                stats = await rollout_actor.get_stats.call_one()
                logger.info(f"RolloutActor stats: {stats}")
                await rollout_actor.shutdown.call_one()
            except Exception:
                pass
        if replay_buffer_actor is not None:
            try:
                stats = await replay_buffer_actor.get_stats.call_one()
                logger.info(f"ReplayBufferActor stats: {stats}")
                await replay_buffer_actor.shutdown.call_one()
            except Exception:
                pass
        if reward_actor is not None:
            try:
                stats = await reward_actor.get_stats.call_one()
                logger.info(f"RewardActor stats: {stats}")
                await reward_actor.shutdown.call_one()
            except Exception:
                pass
        if generator_actor is not None:
            try:
                await generator_actor.shutdown.call_one()
            except Exception:
                pass
        for procs in [rollout_procs, replay_buffer_procs, agent_procs, sandbox_procs, reward_procs, gen_procs]:
            if procs is not None:
                try:
                    procs.stop().get()
                except Exception:
                    pass
        logger.info("MonarchPlugin shutdown complete.")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def monarch_main(config, run_id: int = 0):
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(monarch_main_async(config, run_id))


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    monarch_main(config, run_id=0)


if __name__ == "__main__":
    main()
