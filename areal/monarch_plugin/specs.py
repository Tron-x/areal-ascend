"""Declarative actor specification builder for Monarch orchestration.

Builds the complete list of :class:`ActorSpec` instances from a training
config, allocation mode, and device placement.  This is the single source of
truth for *which* actors get spawned, their resources, dependencies, and
constructor arguments.

The :class:`ActorRegistry` in ``actor_registry.py`` consumes these specs and
handles lifecycle (spawn, init, shutdown) automatically.
"""

from __future__ import annotations

import os
import sys

from areal.api import AllocationType
from areal.api.cli_args import (
    InferenceEngineConfig,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    get_scheduling_spec,
    get_thread_env_vars,
)
from areal.monarch_plugin.actor_spec import ActorRef, ActorSpec, CtxRef, ResourceKind
from areal.monarch_plugin.agent_actor import AgentActor
from areal.monarch_plugin.bootstraps import (
    make_cpu_bootstrap,
    make_generator_bootstrap,
    make_trainer_bootstrap_multi,
    make_trainer_bootstrap_single,
)
from areal.monarch_plugin.executor import WorkerRegistry
from areal.monarch_plugin.generator_actor import GeneratorActor
from areal.monarch_plugin.replay_buffer_actor import ReplayBufferActor
from areal.monarch_plugin.reward_actor import RewardActor
from areal.monarch_plugin.rollout_actor import RolloutActor
from areal.monarch_plugin.sandbox_actor import SandboxActor
from areal.monarch_plugin.trainer_actor import TrainerActor
from areal.monarch_plugin.weight_sync import resolve_xccl_alloc_mode


def build_vllm_cli_args(config, alloc_mode) -> list[str]:
    """Build the CLI arg list for GeneratorActor.

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


def make_actor_specs(
    config,
    alloc_mode,
    placement,
    master_port: int,
    env_var: str,
    inf_device_ids: list[str],
    train_device_ids: list[str],
) -> list[ActorSpec]:
    """Build the list of ActorSpec instances from config.

    This is the *single source of truth* for which actors get spawned,
    their resource requirements, dependencies, and constructor arguments.
    The ActorRegistry consumes these specs and handles lifecycle
    automatically.
    """

    specs: list[ActorSpec] = []

    # --- GeneratorActor ---
    config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
    config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)
    vllm_cli_args = build_vllm_cli_args(config, alloc_mode)

    def _gen_post_spawn(procs, ctx):
        registry = procs.spawn("worker_registry", WorkerRegistry)
        return {"worker_registry": registry}

    specs.append(
        ActorSpec(
            name="generator",
            actor_class=GeneratorActor,
            resource=ResourceKind.NPU_SINGLE,
            bootstrap_factory=lambda: make_generator_bootstrap(
                ",".join(inf_device_ids)
            ),
            constructor_args={"vllm_cli_args": vllm_cli_args},
            dependencies=[],
            init_method="setup",
            init_args={
                "host_mesh": CtxRef("host"),
                "worker_registry": ActorRef("worker_registry"),
                "device_ids": inf_device_ids,
            },
            post_spawn=_gen_post_spawn,
        )
    )

    # --- RewardActor ---
    specs.append(
        ActorSpec(
            name="reward",
            actor_class=RewardActor,
            resource=ResourceKind.CPU,
            bootstrap_factory=make_cpu_bootstrap,
            constructor_args={},
            dependencies=[],
        )
    )

    # --- SandboxActor ---
    specs.append(
        ActorSpec(
            name="sandbox",
            actor_class=SandboxActor,
            resource=ResourceKind.CPU,
            bootstrap_factory=make_cpu_bootstrap,
            constructor_args={},
            dependencies=[],
        )
    )

    # --- AgentActor ---
    specs.append(
        ActorSpec(
            name="agent",
            actor_class=AgentActor,
            resource=ResourceKind.CPU,
            bootstrap_factory=make_cpu_bootstrap,
            constructor_args={
                "generator_actor": ActorRef("generator"),
                "sandbox_actor": ActorRef("sandbox"),
                "reward_actor": ActorRef("reward"),
            },
            dependencies=["generator", "sandbox", "reward"],
        )
    )

    # --- ReplayBufferActor ---
    specs.append(
        ActorSpec(
            name="replay_buffer",
            actor_class=ReplayBufferActor,
            resource=ResourceKind.CPU,
            bootstrap_factory=make_cpu_bootstrap,
            constructor_args={"max_size": 8},
            dependencies=[],
        )
    )

    # --- RolloutActor ---
    specs.append(
        ActorSpec(
            name="rollout",
            actor_class=RolloutActor,
            resource=ResourceKind.CPU,
            bootstrap_factory=make_cpu_bootstrap,
            constructor_args={
                "generator_actor": ActorRef("generator"),
                "reward_actor": ActorRef("reward"),
                "agent_actor": ActorRef("agent"),
            },
            dependencies=["generator", "reward", "agent"],
            init_method="setup",
            init_args={
                "cli_args": sys.argv[1:],
                "train_dp_size": 1,
            },
        )
    )

    # --- TrainerActor (only when training is enabled) ---
    if alloc_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        nprocs = alloc_mode.train.world_size
        multi_rank = nprocs > 1

        actor_sched = get_scheduling_spec(config.actor)
        actor_env_vars = actor_sched.env_vars
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_sched.cpu,
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
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        }

        xccl_alloc_mode = resolve_xccl_alloc_mode(
            config, alloc_mode, train_world_size=nprocs
        )

        if multi_rank:

            def bootstrap_fn():
                return make_trainer_bootstrap_multi(",".join(train_device_ids))
        else:
            _dev_id = int(train_device_ids[0])

            def bootstrap_fn(_d=_dev_id):
                return make_trainer_bootstrap_single(_d)

        specs.append(
            ActorSpec(
                name="trainer",
                actor_class=TrainerActor,
                resource=ResourceKind.NPU_MULTI
                if multi_rank
                else ResourceKind.NPU_SINGLE,
                nprocs=nprocs,
                bootstrap_factory=bootstrap_fn,
                constructor_args={
                    "cli_args": sys.argv[1:],
                    "env_vars": trainer_env,
                    "rank": -1 if multi_rank else 0,
                    "world_size": nprocs,
                    "master_addr": placement.master_addr,
                    "master_port": master_port,
                    "generator_actor": ActorRef("generator"),
                    "reward_actor": ActorRef("reward"),
                    "agent_actor": ActorRef("agent"),
                    "xccl_weight_update_alloc_mode": xccl_alloc_mode,
                },
                dependencies=["generator", "reward", "agent"],
                init_method="initialize",
                is_multi_rank=multi_rank,
                shutdown_broadcast=multi_rank,
            )
        )

    return specs
