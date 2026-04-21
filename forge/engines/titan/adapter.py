"""Forge TrainEngine adapter for TorchTitan's ForgeEngine.

This is the "correct" way to build a training backend for Forge — delegate
all distributed training complexity to TorchTitan (Meta's production-grade
distributed training framework) and only handle the Protocol interface here.

Architecture::

    Forge Protocol (train_step)
         ↓
    TitanTrainEngine (this file, ~200 lines)
         ↓
    TorchTitan ForgeEngine (handles FSDP2/TP/PP/CP/EP, precision, checkpoint)
         ↓
    PyTorch distributed primitives

The adapter is intentionally thin — all precision-sensitive distributed
operations (grad clipping, mixed precision, loss parallel, etc.) are
handled by TorchTitan, not by us.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from forge.core.weight_sync import WeightsSpec

logger = logging.getLogger("TitanEngine")


@dataclass
class TitanConfig:
    """Forge-facing config for the TorchTitan backend."""

    model_name: str = "qwen3"
    model_flavor: str = "1.7B"
    hf_model_path: str = ""
    max_steps: int = 100
    loss_type: str = "grpo"

    lr: float = 1e-6
    dtype: str = "bfloat16"
    seq_len: int = 4096
    local_batch_size: int = 1

    dp_shard: int = -1
    tp: int = 1
    pp: int = 1
    cp: int = 1

    checkpoint_dir: str = "/tmp/forge/titan_checkpoints"
    checkpoint_interval: int = 0

    loss_config: dict[str, Any] = field(default_factory=lambda: {"beta": 0.0})


class TitanTrainEngine:
    """Forge ``TrainEngine`` backed by TorchTitan's ForgeEngine.

    Lifecycle::

        engine = TitanTrainEngine(config={"model_name": "qwen3", "model_flavor": "1.7B"})
        meta = engine.initialize()
        for step in range(meta["max_steps"]):
            result = engine.train_step(batch, step)
        engine.shutdown()
    """

    def __init__(self, config: TitanConfig | dict | None = None, **kwargs):
        if config is None:
            config = TitanConfig(**kwargs)
        elif isinstance(config, dict):
            valid = {f.name for f in TitanConfig.__dataclass_fields__.values()}
            config = TitanConfig(**{k: v for k, v in config.items() if k in valid})
        self._config = config
        self._engine = None
        self._loss_fn = None
        self._current_step = 0
        self._initialized = False

    def initialize(self) -> dict:
        """Create TorchTitan ForgeEngine, build model with full parallelism."""
        import sys

        print("[TitanInit] ENTER initialize()", flush=True)
        sys.stderr.write("[TitanInit] ENTER initialize()\n")
        sys.stderr.flush()

        import torch.distributed as dist

        rank = int(os.environ.get("RANK", -1))
        world = int(os.environ.get("WORLD_SIZE", -1))
        master = os.environ.get("MASTER_ADDR", "")
        port = os.environ.get("MASTER_PORT", "")
        local_rank = os.environ.get("LOCAL_RANK", "")
        import sys

        msg = (
            f"[TitanInit] RANK={rank} WORLD_SIZE={world} "
            f"MASTER_ADDR={master} MASTER_PORT={port} LOCAL_RANK={local_rank} "
            f"dist.is_initialized={dist.is_initialized()}"
        )
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

        if dist.is_initialized():
            print("[TitanInit] dist already initialized, destroying", flush=True)
            dist.destroy_process_group()

        from torchtitan.config.job_config import (
            Checkpoint,
            Job,
            LRScheduler,
            Model,
            Optimizer,
            Parallelism,
            Training,
        )
        from torchtitan.experiments.forge.engine import ForgeEngine
        from torchtitan.experiments.forge.job_config import ForgeJobConfig

        cfg = self._config

        model_kwargs = dict(name=cfg.model_name, flavor=cfg.model_flavor)
        if cfg.hf_model_path:
            model_kwargs["hf_assets_path"] = cfg.hf_model_path

        checkpoint_kwargs = dict(
            folder=cfg.checkpoint_dir,
            interval=cfg.checkpoint_interval,
        )
        if cfg.hf_model_path:
            checkpoint_kwargs.update(
                initial_load_path=cfg.hf_model_path,
                initial_load_in_hf=True,
                initial_load_model_only=True,
            )

        job_config = ForgeJobConfig(
            job=Job(),
            model=Model(**model_kwargs),
            optimizer=Optimizer(lr=cfg.lr),
            lr_scheduler=LRScheduler(),
            training=Training(
                steps=cfg.max_steps,
                dtype=cfg.dtype,
                seq_len=cfg.seq_len,
                local_batch_size=cfg.local_batch_size,
            ),
            parallelism=Parallelism(
                data_parallel_shard_degree=cfg.dp_shard,
                tensor_parallel_degree=cfg.tp,
                pipeline_parallel_degree=cfg.pp,
                context_parallel_degree=cfg.cp,
            ),
            checkpoint=Checkpoint(**checkpoint_kwargs),
        )

        print(f"[TitanInit] rank={rank} creating ForgeEngine...", flush=True)

        self._engine = ForgeEngine(job_config)
        print(
            f"[TitanInit] rank={rank} ForgeEngine created, loading checkpoint...",
            flush=True,
        )

        self._engine.checkpointer.load(step=1)
        print(f"[TitanInit] rank={rank} checkpoint loaded, zero_grad...", flush=True)

        self._engine.optimizers.zero_grad()

        self._loss_fn = self._create_loss()
        self._current_step = 0
        self._initialized = True

        rank = int(os.environ.get("RANK", 0))
        logger.info(
            "TitanTrainEngine[rank=%d] initialized: %s/%s, dp=%d, tp=%d",
            rank,
            cfg.model_name,
            cfg.model_flavor,
            self._engine.dp_degree,
            cfg.tp,
        )

        return {
            "max_steps": cfg.max_steps,
            "start_step": 0,
            "model_path": f"{cfg.model_name}/{cfg.model_flavor}",
        }

    def train_step(self, batch: dict, step: int) -> dict:
        """Run one training step via TorchTitan.

        Follows the same pattern as torchforge's TitanTrainer.train_step:
        forward_backward → all_reduce loss → optimizer step → lr step.
        """
        if not self._initialized:
            raise RuntimeError("TitanTrainEngine not initialized")

        engine = self._engine
        model_parts = engine.model_parts
        device = engine.device

        max_batch = self._config.local_batch_size
        seq_len = self._config.seq_len

        input_ids = self._to_tensor(batch["input_ids"], device, torch.long)
        if input_ids.size(0) > max_batch:
            input_ids = input_ids[:max_batch]
        if input_ids.size(1) > seq_len:
            input_ids = input_ids[:, :seq_len]

        def _truncate(t):
            if t is None:
                return None
            if t.size(0) > max_batch:
                t = t[:max_batch]
            if t.dim() > 1 and t.size(1) > seq_len:
                t = t[:, :seq_len]
            return t

        loss_mask = _truncate(
            self._to_tensor(batch.get("loss_mask"), device)
            if "loss_mask" in batch
            else None
        )
        advantages = _truncate(
            self._to_tensor(batch.get("advantages"), device)
            if "advantages" in batch
            else None
        )
        gen_logprobs = _truncate(
            self._to_tensor(batch.get("generator_logprobs"), device)
            if "generator_logprobs" in batch
            else None
        )

        from forge.rl.loss.ops import create_shifted_targets

        target_ids = create_shifted_targets(input_ids, loss_mask)

        with engine.train_context(None):
            assert len(model_parts) == 1
            with engine.maybe_enable_amp:
                logits = model_parts[0](input_ids)

                if (
                    self._loss_fn
                    and advantages is not None
                    and gen_logprobs is not None
                ):
                    logits_shifted = logits[:, :-1, :]
                    target_shifted = (
                        target_ids[:, :-1] if target_ids.dim() > 1 else target_ids
                    )
                    loss_mask_shifted = (
                        loss_mask[:, 1:] if loss_mask is not None else None
                    )
                    gen_lp_shifted = gen_logprobs[:, 1:]
                    adv_shifted = (
                        advantages[:, 1:] if advantages.dim() > 1 else advantages
                    )

                    loss_output = self._loss_fn(
                        logits=logits_shifted,
                        target_ids=target_shifted,
                        advantages=adv_shifted,
                        generator_logprobs=gen_lp_shifted,
                        loss_mask=loss_mask_shifted
                        if loss_mask_shifted is not None
                        else torch.ones_like(gen_lp_shifted),
                    )
                    loss = loss_output.loss
                else:
                    import torch.nn.functional as F

                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                        ignore_index=-100,
                    )

                del logits
                loss.backward()

        torch.distributed.all_reduce(loss)

        current_lr = engine.lr_schedulers.schedulers[0].get_last_lr()[0]

        engine.optimizers.step()
        engine.optimizers.zero_grad()
        engine.lr_schedulers.step()

        self._current_step = step + 1

        return {
            "loss": loss.detach().item(),
            "lr": current_lr,
            "step": step,
        }

    def get_weights_spec(self) -> WeightsSpec:
        if not self._initialized:
            raise RuntimeError("Not initialized")
        names, shapes, dtypes = [], [], []
        total = 0
        for name, param in self._engine.model_parts[0].named_parameters():
            names.append(name)
            shapes.append(tuple(param.shape))
            dtypes.append(str(param.dtype).replace("torch.", ""))
            total += param.numel()
        return WeightsSpec(
            param_names=names,
            param_shapes=shapes,
            param_dtypes=dtypes,
            total_params=total,
            model_path=f"{self._config.model_name}/{self._config.model_flavor}",
        )

    def state_dict_for_sync(self) -> dict:
        """Return the HF-named full state_dict with tensors on device.

        Three invariants this method must uphold for ``TorchstoreWeightSync``
        + HiXL RoCE + a vLLM generator:

        1. **Tensors stay on the engine's native device (NPU/GPU).**
           HiXL's CANN RA HDC driver rejects CPU-resident memory for
           RoCE MR registration (``ra_hdc_typical_mr ret=-13``, HiXL 排查
           指南 场景二 "HOST 内存…当前不支持注册给 ROCE 网卡"), and
           torchstore's ``MonarchRDMATransportBuffer.allocate`` skips its
           NPU staging pool for CPU tensors -- so ``.cpu()`` here would
           route straight into the failure path.
        2. **Tensors are plain ``torch.Tensor``, not ``DTensor``.**
           FSDP2's ``model.state_dict()`` returns DTensors whose shards
           are distributed across ranks; copying into a plain flat buffer
           raises ``aten.copy_.default: got mixed torch.Tensor and
           DTensor``.  We materialise each DTensor to a replicated full
           tensor via ``.full_tensor()`` (a collective, invoked
           identically on every rank).
        3. **Parameter names are in HuggingFace form, not TorchTitan
           native.**  TorchTitan's ``Qwen3`` model exposes names like
           ``tok_embeddings.weight``, ``norm.weight``, ...; the vLLM
           generator, on the other hand, expects HF-standard names
           (``model.embed_tokens.weight``, ``lm_head.weight``, ...).
           Without the rename, vLLM's ``load_weights`` path raises
           ``ValueError: There is no module or parameter named
           'tok_embeddings' in Qwen3ForCausalLM``.  We apply the
           checkpointer's ``sd_adapter.to_hf(...)`` to do the mapping
           (the exact flow upstream torchforge uses in
           ``src/forge/actors/trainer/titan.py::push_weights``).
        """
        if not self._initialized:
            raise RuntimeError("Not initialized")
        from torch.distributed.checkpoint._nested_dict import flatten_state_dict
        from torch.distributed.tensor import DTensor

        # Collect the TorchTitan-native state_dict straight off the
        # (possibly sharded) model.  We deliberately do *not* go through
        # ``self._engine.checkpointer.states["model"]`` the way upstream
        # torchforge does: our ForgeEngineConfig leaves
        # ``checkpoint.enable=False`` by default (we don't persist
        # intermediate weights to disk), and CheckpointManager.__init__
        # early-returns in that case without populating ``self.states``
        # or ``self.sd_adapter`` -- so accessing either raises
        # ``AttributeError``.  Reading the model's state_dict directly
        # gives us the same tensors with none of that coupling.
        raw = self._engine.model_parts[0].state_dict()
        flat, _spec = flatten_state_dict(raw)

        sd_adapter = self._get_sd_adapter()
        if sd_adapter is None:
            raise RuntimeError(
                "TitanTrainEngine.state_dict_for_sync: no sd_adapter is "
                "available for this model (train_spec.state_dict_adapter "
                "is None).  Without it we cannot translate "
                "TorchTitan-native parameter names into the HF form vLLM "
                "expects ('no module or parameter named <tt-native>' on "
                "the generator side)."
            )
        hf_sd = sd_adapter.to_hf(flat)

        out: dict = {}
        for k, v in hf_sd.items():
            if isinstance(v, DTensor):
                v = v.full_tensor()
            out[k] = v
        return out

    def _get_sd_adapter(self):
        """Lazily build (and cache) the TorchTitan StateDictAdapter for
        this engine's model.

        The adapter is what ``CheckpointManager`` would have constructed
        if checkpointing were enabled, built from
        ``self._engine.train_spec.state_dict_adapter`` +
        ``self._engine.model_args`` + ``self._config.hf_model_path``.
        """
        existing = getattr(self, "_sd_adapter", None)
        if existing is not None:
            return existing
        train_spec = getattr(self._engine, "train_spec", None)
        if (
            train_spec is None
            or getattr(train_spec, "state_dict_adapter", None) is None
        ):
            return None
        adapter = train_spec.state_dict_adapter(
            self._engine.model_args, self._config.hf_model_path or ""
        )
        self._sd_adapter = adapter
        return adapter

    def get_metadata(self) -> dict:
        return {
            "max_steps": self._config.max_steps,
            "current_step": self._current_step,
            "model_name": self._config.model_name,
            "model_flavor": self._config.model_flavor,
            "backend": "titan",
        }

    def shutdown(self) -> None:
        logger.info("TitanTrainEngine shutting down")
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._initialized = False

    def _create_loss(self):
        loss_kwargs = self._config.loss_config
        if self._config.loss_type == "grpo":
            from forge.rl.loss import GRPOLoss

            return GRPOLoss(**loss_kwargs)
        elif self._config.loss_type == "dapo":
            from forge.rl.loss import DAPOLoss

            return DAPOLoss(**loss_kwargs)
        return None

    @staticmethod
    def _to_tensor(data, device, dtype=None):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            t = data.to(device=device)
            return t.to(dtype=dtype) if dtype else t
        return torch.tensor(data, device=device, dtype=dtype)
