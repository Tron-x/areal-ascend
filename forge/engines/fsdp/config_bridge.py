"""FSDP config bridge -- standalone config parsing without AReaL dependency.

Provides a ``ConfigBridge`` that builds ``ForgeConfig`` from simple CLI
arguments or a YAML file, bypassing AReaL's config system entirely.

Usage::

    bridge = FSDPConfigBridge()
    forge_cfg, raw_cfg, alloc_mode = bridge.parse_and_build()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


class FSDPConfigBridge:
    """Standalone config bridge for the FSDP training backend.

    Parses CLI arguments directly (no AReaL dependency) and produces
    a ``ForgeConfig`` suitable for the FSDP ``TrainEngine`` path.
    """

    def parse_and_build(
        self,
        argv: list[str] | None = None,
        run_id: int = 0,
    ) -> tuple[Any, dict, None]:
        """Parse CLI args and return (ForgeConfig, raw_config_dict, None).

        The third element is None (no AllocationMode concept in standalone).
        """
        from forge.core.config import ForgeConfig

        if argv is None:
            argv = sys.argv[1:]

        args, extra = self._parse_args(argv)

        raw_cfg = vars(args)
        for item in extra:
            if "=" in item:
                key, val = item.split("=", 1)
                key = key.lstrip("+").strip()
                raw_cfg[key] = _auto_cast(val)

        fileroot = args.fileroot
        os.makedirs(fileroot, exist_ok=True)

        from forge.utils.network import find_free_ports, gethostip

        master_addr = gethostip()
        master_port = find_free_ports(1, (10000, 50000))[0]

        engine_args = {
            "model": args.model,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": True,
            "seed": args.seed,
        }

        # FSDP path currently assumes TP=PP=1 on both sides (no DSL).
        # If we ever add CLI knobs for TP/PP here, replace these
        # defaults with the parsed values; see the areal bridge for
        # the alloc_mode-DSL-driven version.
        forge_cfg = ForgeConfig(
            experiment_name=args.experiment_name,
            trial_name=args.trial_name,
            run_id=run_id,
            model_path=args.model,
            train_world_size=args.train_gpus,
            gen_world_size=args.gen_gpus,
            gen_dp_size=args.gen_gpus,
            gen_tp_size=1,
            gen_pp_size=1,
            train_dp_size=args.train_gpus,
            train_tp_size=1,
            train_pp_size=1,
            master_addr=master_addr,
            master_port=master_port,
            reward_fn_path=args.reward_fn,
            training_script="",
            training_args=[],
            engine_args=engine_args,
            trainer_env=self._build_trainer_env(),
            backend_type="fsdp",
            backend_config={
                "model_path": args.model,
                "loss_type": args.loss_type,
                "max_steps": args.total_train_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "warmup_steps": args.warmup_steps,
                "dtype": args.dtype,
                "loss_config": {
                    "clip_low": args.clip_low,
                    "clip_high": args.clip_high,
                    "beta": args.kl_coef,
                },
            },
            fileroot=fileroot,
        )

        return forge_cfg, raw_cfg, None

    def setup_name_resolve(self, raw_cfg: dict) -> None:
        """No-op for standalone FSDP (no AReaL name resolution)."""
        nfs_root = os.path.join(
            raw_cfg.get("fileroot", "/tmp/forge"),
            "name_resolve",
        )
        os.makedirs(nfs_root, exist_ok=True)

    def save_metadata(self, raw_cfg: dict) -> None:
        """Save minimal experiment metadata."""
        import json

        fileroot = raw_cfg.get("fileroot", "/tmp/forge")
        meta_dir = os.path.join(
            fileroot,
            raw_cfg.get("experiment_name", "exp"),
            raw_cfg.get("trial_name", "trial0"),
        )
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({k: str(v) for k, v in raw_cfg.items()}, f, indent=2)

    @staticmethod
    def is_llm_server_only(alloc_mode) -> bool:
        return False

    @staticmethod
    def resolve_xccl_alloc_mode(raw_cfg, alloc_mode, train_world_size: int):
        return None

    @staticmethod
    def _build_trainer_env() -> dict[str, str]:
        return {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        }

    @staticmethod
    def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
        p = argparse.ArgumentParser(
            description="Forge FSDP Training",
            allow_abbrev=False,
        )

        p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
        p.add_argument("--experiment-name", default="fsdp-grpo")
        p.add_argument("--trial-name", default="trial0")

        p.add_argument("--train-gpus", type=int, default=4)
        p.add_argument("--gen-gpus", type=int, default=4)
        p.add_argument("--total-train-steps", type=int, default=100)

        p.add_argument("--loss-type", default="grpo", choices=["grpo", "dapo"])
        p.add_argument("--lr", type=float, default=1e-6)
        p.add_argument("--weight-decay", type=float, default=0.01)
        p.add_argument("--max-grad-norm", type=float, default=1.0)
        p.add_argument("--warmup-steps", type=int, default=10)
        p.add_argument("--clip-low", type=float, default=0.2)
        p.add_argument("--clip-high", type=float, default=0.28)
        p.add_argument("--kl-coef", type=float, default=0.0)

        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--max-model-len", type=int, default=4096)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
        p.add_argument("--enforce-eager", action="store_true", default=True)
        p.add_argument("--seed", type=int, default=42)

        p.add_argument(
            "--reward-fn", default="forge.examples.harbor.reward.harbor_math_reward"
        )
        p.add_argument("--fileroot", default="/tmp/forge/experiments")

        return p.parse_known_args(argv)


def _auto_cast(val: str) -> Any:
    """Try to cast string to int/float/bool."""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
