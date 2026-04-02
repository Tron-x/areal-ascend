"""In-process FSDP training actor for Monarch orchestration.

TrainerActor runs the full AReaL training pipeline (PPOTrainer) inside a
Monarch ProcMesh.  It uses the ``trainer_hooks`` extension points to inject
MonarchVLLMEngine and override XCCL alloc_mode — **no monkey-patching**.
"""

from __future__ import annotations

import importlib.util
import os

from monarch.actor import endpoint

from areal.api import AllocationMode
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    get_scheduling_spec,
    get_thread_env_vars,
)
from areal.monarch_plugin.actor_base import MonarchActor
from areal.monarch_plugin.actor_spec import ActorRef, ResourceKind
from areal.utils import logging
from areal.utils.trainer_hooks import clear_hooks, set_hooks

logger = logging.getLogger("TrainerActor")


class TrainerActor(MonarchActor):
    """In-process FSDP training actor with Monarch RPC rollout and reward.

    Uses ``trainer_hooks`` to inject ``MonarchVLLMEngine`` and override
    XCCL weight-update ``alloc_mode`` instead of monkey-patching PPOTrainer
    or FSDPEngine.
    """

    dependencies = ["generator", "reward", "agent"]

    @classmethod
    def resolve_resource(cls, ctx) -> ResourceKind:
        nprocs = ctx.alloc_mode.train.world_size
        return ResourceKind.NPU_MULTI if nprocs > 1 else ResourceKind.NPU_SINGLE

    @classmethod
    def resolve_nprocs(cls, ctx) -> int:
        return ctx.alloc_mode.train.world_size

    @classmethod
    def resolve_is_multi_rank(cls, ctx) -> bool:
        return ctx.alloc_mode.train.world_size > 1

    @classmethod
    def resolve_shutdown_broadcast(cls, ctx) -> bool:
        return cls.resolve_is_multi_rank(ctx)

    @classmethod
    def init_method(cls) -> str | None:
        return "initialize"

    @classmethod
    def bootstrap_factory(cls, ctx):
        from areal.monarch_plugin.bootstraps import (
            make_trainer_bootstrap_multi,
            make_trainer_bootstrap_single,
        )

        train_device_ids = ctx.placement.training.all_device_ids
        nprocs = ctx.alloc_mode.train.world_size
        if nprocs > 1:
            return make_trainer_bootstrap_multi(",".join(train_device_ids))
        else:
            return make_trainer_bootstrap_single(int(train_device_ids[0]))

    @classmethod
    def constructor_args(cls, ctx) -> dict:
        import os
        import sys

        from areal.monarch_plugin.weight_sync import resolve_xccl_alloc_mode

        env_var = ctx.extra.get("env_var", "ASCEND_RT_VISIBLE_DEVICES")
        train_device_ids = ctx.placement.training.all_device_ids
        nprocs = ctx.alloc_mode.train.world_size
        multi_rank = nprocs > 1

        actor_sched = get_scheduling_spec(ctx.config.actor)
        actor_env_vars = actor_sched.env_vars
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_sched.cpu,
            existing_env_vars=actor_env_vars,
        )
        tms_env_vars = {}
        if ctx.config.get("enable_offload", False):
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
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        }

        xccl_alloc_mode = resolve_xccl_alloc_mode(
            ctx.config, ctx.alloc_mode, train_world_size=nprocs
        )

        return {
            "cli_args": sys.argv[1:],
            "env_vars": trainer_env,
            "rank": -1 if multi_rank else 0,
            "world_size": nprocs,
            "master_addr": ctx.placement.master_addr,
            "master_port": ctx.master_port,
            "generator_actor": ActorRef("generator"),
            "reward_actor": ActorRef("reward"),
            "agent_actor": ActorRef("agent"),
            "xccl_weight_update_alloc_mode": xccl_alloc_mode,
            "train_dp_size": ctx.alloc_mode.train.dp_size,
        }

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
        train_dp_size: int = 1,
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
        self._train_dp_size = train_dp_size
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

        # -- Build the rollout engine factory closure -----------------------
        generator_ref = self._generator
        reward_ref = self._reward
        agent_ref = self._agent

        def _monarch_rollout_factory(rollout_config, is_eval=False, lora_path=None):
            """Create MonarchVLLMEngine instead of the default RemotevLLMEngine."""
            from areal.monarch_plugin.monarch_inf_engine import MonarchVLLMEngine

            engine = MonarchVLLMEngine(
                rollout_config,
                generator_ref,
                reward_actor=reward_ref,
                agent_actor=agent_ref,
            )
            engine.initialize(
                train_data_parallel_size=self._train_dp_size,
            )
            logger.info(
                f"TrainerActor: created MonarchVLLMEngine "
                f"(is_eval={is_eval}, reward_actor={'yes' if reward_ref else 'no'}, "
                f"agent_actor={'yes' if agent_ref else 'no'})"
            )
            return engine

        # -- Install trainer hooks ------------------------------------------
        set_hooks(
            rollout_engine_factory=_monarch_rollout_factory,
            weight_update_alloc_mode_override=self._xccl_weight_update_alloc_mode,
            suppress_context_exit=True,
        )

        # Set class-level flag so train() bails out immediately.
        PPOTrainer._skip_train_loop = True

        _captured: dict = {}
        _orig_train = PPOTrainer.train

        def _intercept_train(trainer_self, **kwargs):
            _captured["trainer"] = trainer_self
            _captured["kwargs"] = kwargs

        PPOTrainer.train = _intercept_train

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
            PPOTrainer._skip_train_loop = False
            clear_hooks()

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
    # Training endpoints — delegate directly to PPOTrainer.run_single_step
    # ------------------------------------------------------------------

    @endpoint
    def train_step(self, global_step: int) -> dict:
        """Run one GRPO/PPO training iteration (rollout + train combined)."""
        return self._trainer.run_single_step(global_step, **self._train_kwargs)

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        """Run rollout only, serialise the batch for ReplayBuffer."""
        return self._trainer.run_single_step(global_step, **self._train_kwargs)

    @endpoint
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict:
        """Train on a pre-produced rollout batch from the ReplayBuffer."""
        return self._trainer.run_single_step(
            global_step, rollout_batch=batch_data, **self._train_kwargs
        )

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
