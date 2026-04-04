"""AReaL config bridge — translates AReaL CLI args into ForgeConfig.

This is the ONLY place in forge that imports ``areal.api.cli_args``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


class AReaLConfigBridge:
    """Translates AReaL's CLI config into Forge's config structures.

    Usage::

        bridge = AReaLConfigBridge()
        forge_cfg, raw_cfg, alloc_mode = bridge.parse_and_build()
    """

    def parse_and_build(
        self,
        argv: list[str] | None = None,
        run_id: int = 0,
    ) -> tuple[Any, Any, Any]:
        """Parse CLI args and return (ForgeConfig, raw_areal_config, AllocationMode).

        The raw_areal_config is the OmegaConf DictConfig from ``parse_cli_args``,
        needed by adapter internals that still require the full AReaL config tree.
        """
        from forge.core.config import ForgeConfig

        from areal.api import AllocationMode
        from areal.api.cli_args import (
            ClusterSpecConfig,
            InferenceEngineConfig,
            RecoverConfig,
            parse_cli_args,
            to_structured_cfg,
            vLLMConfig,
        )
        from areal.infra.utils.launcher import validate_config_for_launcher
        from areal.utils.recover import check_if_recover

        if argv is None:
            argv = sys.argv[1:]

        config, _ = parse_cli_args(argv)

        config.recover = to_structured_cfg(config.recover, RecoverConfig)
        config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
        is_recover_run = check_if_recover(config.recover, run_id)
        validate_config_for_launcher(config)

        config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
        config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)

        alloc_mode = AllocationMode.from_str(config.allocation_mode)

        train_ws = alloc_mode.train.world_size if alloc_mode.train else 0

        from forge.utils.network import find_free_ports, gethostip

        master_addr = gethostip()
        master_port = find_free_ports(1, (10000, 50000))[0]

        engine_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=alloc_mode.gen.tp_size,
            pp_size=alloc_mode.gen.pp_size,
        )

        trainer_env = self._build_trainer_env(config)

        forge_cfg = ForgeConfig(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            run_id=run_id,
            model_path=config.vllm.model,
            train_world_size=train_ws,
            gen_world_size=alloc_mode.gen.world_size,
            master_addr=master_addr,
            master_port=master_port,
            reward_fn_path=(
                config.get("reward_fn") or "areal.reward.gsm8k.gsm8k_reward_fn"
            ),
            training_script=argv[0] if argv else "",
            training_args=argv,
            engine_args=engine_args,
            trainer_env=trainer_env,
            backend_type="areal",
            backend_config={
                "is_recover_run": is_recover_run,
            },
            fileroot=config.cluster.fileroot,
        )

        return forge_cfg, config, alloc_mode

    @staticmethod
    def setup_name_resolve(config) -> None:
        """Initialize AReaL's name resolution service."""
        from areal.utils import name_resolve, names

        name_resolve.reconfigure(config.cluster.name_resolve)
        name_resolve.clear_subtree(
            names.trial_root(
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
            )
        )

    @staticmethod
    def save_metadata(config) -> None:
        """Save experiment metadata to disk."""
        from areal.infra.utils.exp_metadata import save_experiment_metadata

        save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )

    @staticmethod
    def resolve_xccl_alloc_mode(config, alloc_mode, train_world_size: int):
        """Resolve XCCL weight-update allocation mode."""
        from forge.engines.areal.weight_sync import resolve_xccl_alloc_mode

        return resolve_xccl_alloc_mode(
            config, alloc_mode, train_world_size=train_world_size
        )

    @staticmethod
    def is_llm_server_only(alloc_mode) -> bool:
        from areal.api import AllocationType

        return alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY

    @staticmethod
    def _build_trainer_env(config) -> dict[str, str]:
        from areal.infra.utils.launcher import (
            BASE_ENVIRONS,
            get_scheduling_spec,
            get_thread_env_vars,
        )

        actor_spec = get_scheduling_spec(config.actor)
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_spec.cpu,
            existing_env_vars=actor_spec.env_vars,
        )
        return {
            **BASE_ENVIRONS,
            **thread_env,
            **actor_spec.env_vars,
            "AREAL_SPMD_MODE": "1",
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        }
