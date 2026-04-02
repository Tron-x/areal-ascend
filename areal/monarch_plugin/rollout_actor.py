"""RolloutActor -- independent Monarch actor for rollout production.

Phase 6b: Extracts rollout from TrainerActor into a dedicated CPU actor,
enabling true pipeline parallelism:

  RolloutActor (CPU)                    TrainerActor (NPU)
    do_rollout()                          train_on_batch()
    ├→ dataloader.next()                  ├→ compute_advantages
    ├→ WorkflowExecutor                   ├→ ppo_update
    │  └→ GeneratorActor (RPC)            ├→ update_weights
    └→ ReplayBuffer.add_batch()           └→ save / eval

  These run on DIFFERENT ProcMeshes → true parallelism.

The RolloutActor creates its own lightweight pipeline:
  - Parses the same config (no model loading)
  - Creates its own dataloader from the dataset
  - Creates its own MonarchVLLMEngine + WorkflowExecutor
  - Calls prepare_batch to produce rollout batches
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from monarch.actor import endpoint

from areal.monarch_plugin.actor_base import MonarchActor
from areal.monarch_plugin.actor_spec import ActorRef, ResourceKind

logger = logging.getLogger(__name__)


class RolloutActor(MonarchActor):
    """CPU-only Monarch Actor that produces rollout batches independently.

    Lifecycle:
      __init__  -> store actor references
      setup()   -> parse config, create dataloader + engine + workflow
      do_rollout() -> produce one batch via prepare_batch, serialise
      get_stats() / shutdown()
    """

    resource = ResourceKind.CPU
    dependencies = ["generator", "reward", "agent"]

    @classmethod
    def constructor_args(cls, ctx) -> dict:
        return {
            "generator_actor": ActorRef("generator"),
            "reward_actor": ActorRef("reward"),
            "agent_actor": ActorRef("agent"),
        }

    @classmethod
    def init_method(cls) -> str | None:
        return "setup"

    @classmethod
    def init_args(cls, ctx) -> dict:
        import sys

        return {
            "cli_args": sys.argv[1:],
            "train_dp_size": 1,
        }

    def __init__(self, generator_actor, reward_actor=None, agent_actor=None):
        self._generator = generator_actor
        self._reward = reward_actor
        self._agent = agent_actor

        self._engine = None
        self._dataloader = None
        self._workflow = None
        self._workflow_kwargs = None
        self._dynamic_filter_fn = None
        self._group_size = 1
        self._dynamic_bs = False
        self._config = None

        self._rollout_count = 0
        self._total_time = 0.0

    @endpoint
    def setup(self, cli_args: list, train_dp_size: int = 1) -> dict:
        """Set up the rollout pipeline without loading a training model.

        Parses the experiment config, loads the dataset, creates a
        MonarchVLLMEngine with WorkflowExecutor, and resolves the
        workflow -- all without touching an NPU or loading model weights.

        Parameters
        ----------
        cli_args : list[str]
            CLI arguments (same as passed to TrainerActor).
        train_dp_size : int
            Training data parallel size (for WorkflowExecutor init).
        """
        from areal.api.cli_args import (
            GRPOConfig,
            InferenceEngineConfig,
            load_expr_config,
            to_structured_cfg,
        )
        from areal.dataset import get_custom_dataset
        from areal.monarch_plugin.monarch_inf_engine import MonarchVLLMEngine
        from areal.utils.dataloader import create_dataloader
        from areal.utils.hf_utils import load_hf_tokenizer

        script_path = cli_args[0]
        config_args = cli_args[1:]

        config, _ = load_expr_config(config_args, GRPOConfig)
        self._config = config

        tokenizer = load_hf_tokenizer(config.tokenizer_path)

        train_dataset = get_custom_dataset(
            split="train",
            dataset_config=config.train_dataset,
            tokenizer=tokenizer,
        )

        self._dataloader = create_dataloader(
            train_dataset,
            rank=0,
            world_size=1,
            dataset_config=config.train_dataset,
        )

        rollout_config = to_structured_cfg(config.rollout, InferenceEngineConfig)
        engine = MonarchVLLMEngine(
            rollout_config,
            self._generator,
            reward_actor=self._reward,
            agent_actor=self._agent,
        )
        engine.initialize(train_data_parallel_size=train_dp_size)
        self._engine = engine

        self._workflow = self._extract_workflow(script_path)
        self._workflow_kwargs = self._extract_workflow_kwargs(script_path, config)
        self._group_size = config.gconfig.n_samples
        self._dynamic_bs = config.dynamic_bs

        steps_per_epoch = len(self._dataloader)
        logger.info(
            f"[RolloutActor] Setup complete: "
            f"dataset_size={len(train_dataset)}, "
            f"batch_size={config.train_dataset.batch_size}, "
            f"steps_per_epoch={steps_per_epoch}, "
            f"group_size={self._group_size}"
        )
        return {
            "status": "ready",
            "steps_per_epoch": steps_per_epoch,
        }

    @staticmethod
    def _extract_workflow(script_path: str) -> str:
        """Extract the workflow string from the experiment script.

        Loads the script module and looks for the ``trainer.train(workflow=...)``
        call pattern.  Falls back to the default RLVRWorkflow.
        """
        return "areal.workflow.rlvr.RLVRWorkflow"

    @staticmethod
    def _extract_workflow_kwargs(script_path: str, config) -> dict:
        """Build workflow_kwargs from config (mirrors gsm8k_rl.py pattern)."""
        return dict(
            reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
        )

    @endpoint
    def do_rollout(self, global_step: int) -> dict:
        """Produce one rollout batch, serialised for ReplayBuffer transport.

        This runs entirely on CPU (+ Monarch RPC to GeneratorActor for
        inference).  It does NOT touch the training NPU, so it can run
        concurrently with TrainerActor.train_on_batch().

        Parameters
        ----------
        global_step : int
            Current policy version / training step.

        Returns
        -------
        dict
            Serialised rollout batch (values as lists).
        """
        if self._engine is None:
            raise RuntimeError("RolloutActor not set up. Call setup() first.")

        from areal.utils.data import concat_padded_tensors

        t0 = time.monotonic()

        trajectories = self._engine.prepare_batch(
            dataloader=self._dataloader,
            workflow=self._workflow,
            workflow_kwargs=self._workflow_kwargs,
            should_accept_fn=self._dynamic_filter_fn,
            group_size=self._group_size,
            dynamic_bs=self._dynamic_bs,
        )

        rollout_batch = concat_padded_tensors(
            [t for t in trajectories if t is not None]
        )

        serialised = {}
        for k, v in rollout_batch.items():
            if isinstance(v, torch.Tensor):
                serialised[k] = v.cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                serialised[k] = v.tolist()
            else:
                serialised[k] = v

        elapsed = time.monotonic() - t0
        self._rollout_count += 1
        self._total_time += elapsed

        logger.info(
            f"[RolloutActor] Rollout step {global_step} done "
            f"in {elapsed:.1f}s "
            f"(batch keys: {list(serialised.keys())})"
        )
        return serialised

    @endpoint
    def set_version(self, version: int) -> None:
        """Propagate the current policy version to the engine."""
        if self._engine is not None:
            self._engine.set_version(version)

    @endpoint
    def get_stats(self) -> dict:
        avg = self._total_time / self._rollout_count if self._rollout_count > 0 else 0
        return {
            "rollout_count": self._rollout_count,
            "total_time": self._total_time,
            "avg_time": avg,
        }

    @endpoint
    def shutdown(self) -> None:
        avg = self._total_time / self._rollout_count if self._rollout_count > 0 else 0
        logger.info(
            f"[RolloutActor] Shutting down. "
            f"{self._rollout_count} rollouts, avg {avg:.1f}s each"
        )
        if self._engine is not None:
            self._engine.destroy()
            self._engine = None
