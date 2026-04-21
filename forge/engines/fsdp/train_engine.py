"""FSDPTrainEngine — native PyTorch FSDP2 training engine.

Implements the ``TrainEngine`` protocol using PyTorch's FSDP2
(``torch.distributed.fsdp.fully_shard``) for distributed training.
This is the first non-AReaL training backend for Forge.

Designed for GRPO/DAPO-style policy gradient training with:
- HuggingFace model loading (``AutoModelForCausalLM``)
- FSDP2 ``fully_shard`` wrapping (no legacy ``FullyShardedDataParallel``)
- AdamW optimizer with cosine LR schedule
- Gradient clipping
- Forge's ``GRPOLoss`` / ``DAPOLoss`` for policy gradient

Usage::

    engine = FSDPTrainEngine(
        model_path="Qwen/Qwen2.5-1.5B",
        loss_type="grpo",
        max_steps=100,
    )
    meta = engine.initialize()
    for step in range(meta["max_steps"]):
        result = engine.train_step(batch, step)
    engine.shutdown()

Note: This engine expects an **already-initialized** torch.distributed
process group.  The ``TrainerActor`` / ``Provisioner`` handles process
group setup before calling ``initialize()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from forge.core.weight_sync import WeightsSpec

logger = logging.getLogger("FSDPTrainEngine")


@dataclass
class FSDPTrainEngineConfig:
    """Configuration for FSDPTrainEngine.

    Attributes:
        model_path: HuggingFace model ID or local path.
        loss_type: Loss function (``"grpo"`` or ``"dapo"``).
        max_steps: Maximum training steps.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping norm.
        warmup_steps: LR warmup steps.
        dtype: Model dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        loss_config: Extra kwargs for the loss function.
    """

    model_path: str = ""
    loss_type: str = "grpo"
    max_steps: int = 100
    lr: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    dtype: str = "bfloat16"
    loss_config: dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str = ""
    save_every: int = 0


class FSDPTrainEngine:
    """Native FSDP2 training engine implementing ``TrainEngine`` protocol.

    This engine wraps a HuggingFace causal LM with PyTorch FSDP2 and
    provides the standard ``TrainEngine`` lifecycle.
    """

    def __init__(self, config: FSDPTrainEngineConfig | dict | None = None, **kwargs):
        if config is None:
            config = FSDPTrainEngineConfig(**kwargs)
        elif isinstance(config, dict):
            config = FSDPTrainEngineConfig(**config)
        self._config = config
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._loss_fn = None
        self._current_step = 0
        self._initialized = False

    def initialize(self) -> dict:
        """Load model, wrap with FSDP2, create optimizer and scheduler.

        Also initializes ``torch.distributed`` if not already done
        (standalone mode without AReaL).
        """
        import torch
        import torch.distributed as dist

        self._init_distributed()

        cfg = self._config
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

        logger.info("Loading model: %s (dtype=%s)", cfg.model_path, cfg.dtype)
        self._model = self._load_model(cfg.model_path, torch_dtype)
        self._wrap_fsdp()

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self._scheduler = self._create_scheduler()
        self._loss_fn = self._create_loss()

        self._current_step = 0
        self._initialized = True

        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(
            "FSDPTrainEngine[rank=%d] initialized: max_steps=%d",
            rank,
            cfg.max_steps,
        )
        return {
            "max_steps": cfg.max_steps,
            "start_step": 0,
            "model_path": cfg.model_path,
        }

    def train_step(self, batch: dict, step: int) -> dict:
        """Run one GRPO/DAPO training step.

        Expected batch keys (from ``BatchAdapter``):
            - ``input_ids``: [B, seq_len] token ids
            - ``attention_mask``: [B, seq_len]
            - ``loss_mask``: [B, seq_len] mask for response tokens
            - ``advantages``: [B, seq_len] or [B] advantages
            - ``generator_logprobs``: [B, seq_len] old policy log probs
            - ``ref_logprobs``: [B, seq_len] reference log probs (optional)

        Returns:
            Dict with ``loss``, ``grad_norm``, ``lr``.
        """
        import torch

        if not self._initialized:
            raise RuntimeError("FSDPTrainEngine not initialized")

        self._model.train()
        self._optimizer.zero_grad()

        device = next(self._model.parameters()).device
        input_ids = self._to_tensor(batch["input_ids"], device, torch.long)
        attention_mask = self._to_tensor(batch["attention_mask"], device, torch.long)
        loss_mask = self._to_tensor(batch["loss_mask"], device)
        advantages = self._to_tensor(batch["advantages"], device)
        gen_logprobs = self._to_tensor(batch["generator_logprobs"], device)
        ref_logprobs = (
            self._to_tensor(batch["ref_logprobs"], device)
            if "ref_logprobs" in batch
            else None
        )

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        target_ids = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        loss_mask_shifted = loss_mask[:, 1:]
        gen_logprobs_shifted = gen_logprobs[:, 1:]
        advantages_shifted = advantages[:, 1:] if advantages.dim() > 1 else advantages
        ref_lp_shifted = ref_logprobs[:, 1:] if ref_logprobs is not None else None

        loss_output = self._loss_fn(
            logits=logits,
            target_ids=target_ids,
            advantages=advantages_shifted,
            generator_logprobs=gen_logprobs_shifted,
            loss_mask=loss_mask_shifted,
            ref_logprobs=ref_lp_shifted,
        )

        loss_output.loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), self._config.max_grad_norm
        )

        self._optimizer.step()
        self._scheduler.step()

        self._current_step = step + 1

        return {
            "loss": loss_output.loss.item(),
            "grad_norm": grad_norm.item()
            if hasattr(grad_norm, "item")
            else float(grad_norm),
            "lr": self._scheduler.get_last_lr()[0],
            "step": step,
        }

    def get_weights_spec(self) -> WeightsSpec:
        """Return a description of model parameters for weight sync."""
        if not self._initialized:
            raise RuntimeError("FSDPTrainEngine not initialized")

        names = []
        shapes = []
        dtypes = []
        total = 0
        for name, param in self._model.named_parameters():
            names.append(name)
            shapes.append(tuple(param.shape))
            dtypes.append(str(param.dtype).replace("torch.", ""))
            total += param.numel()

        return WeightsSpec(
            param_names=names,
            param_shapes=shapes,
            param_dtypes=dtypes,
            total_params=total,
            model_path=self._config.model_path,
        )

    def state_dict_for_sync(self) -> dict:
        """Return the model state dict for weight sync.

        For FSDP2 models, this returns the local shard. The
        ``WeightSyncStrategy`` handles reassembly if needed.

        Tensors are kept on device (NPU/GPU): HiXL RDMA cannot register
        CPU-resident memory (``ra_hdc_typical_mr ret=-13``).  Any
        downstream consumer that really needs CPU should do ``.cpu()``
        itself after the RDMA transfer completes -- see
        ``forge/engines/titan/adapter.py::state_dict_for_sync`` for the
        full root-cause writeup.
        """
        if not self._initialized:
            raise RuntimeError("FSDPTrainEngine not initialized")
        return dict(self._model.state_dict().items())

    def get_metadata(self) -> dict:
        return {
            "max_steps": self._config.max_steps,
            "current_step": self._current_step,
            "model_path": self._config.model_path,
            "loss_type": self._config.loss_type,
            "dtype": self._config.dtype,
        }

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save model checkpoint to disk.

        Args:
            path: Directory to save to. If None, uses config checkpoint_dir.

        Returns:
            Path where checkpoint was saved.
        """
        import os

        import torch
        import torch.distributed as dist

        if not self._initialized:
            raise RuntimeError("FSDPTrainEngine not initialized")

        save_dir = path or self._config.checkpoint_dir
        if not save_dir:
            save_dir = f"/tmp/forge/checkpoints/step_{self._current_step}"
        os.makedirs(save_dir, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            state_dict = get_model_state_dict(
                self._model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            state_dict = self._model.state_dict()

        if rank == 0:
            ckpt_path = os.path.join(save_dir, "model.safetensors")
            try:
                from safetensors.torch import save_file

                save_file(state_dict, ckpt_path)
            except ImportError:
                ckpt_path = os.path.join(save_dir, "pytorch_model.bin")
                torch.save(state_dict, ckpt_path)

            meta = {
                "step": self._current_step,
                "model_path": self._config.model_path,
                "loss_type": self._config.loss_type,
                "lr": self._scheduler.get_last_lr()[0],
            }
            import json

            with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                json.dump(meta, f, indent=2)

            logger.info("Checkpoint saved: %s (step %d)", save_dir, self._current_step)

        if dist.is_initialized():
            dist.barrier()

        return save_dir

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint from disk.

        Args:
            path: Directory containing the checkpoint.

        Returns:
            The training step at which the checkpoint was saved.
        """
        import json
        import os

        import torch

        if not self._initialized:
            raise RuntimeError("FSDPTrainEngine not initialized")

        safetensors_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No checkpoint found in {path}")

        self._model.load_state_dict(state_dict, strict=False)

        meta_path = os.path.join(path, "training_state.json")
        step = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            step = meta.get("step", 0)

        self._current_step = step
        logger.info("Checkpoint loaded from %s (step %d)", path, step)
        return step

    def shutdown(self) -> None:
        logger.info("FSDPTrainEngine shutting down")
        if self._model is not None:
            del self._model
            self._model = None
        if self._optimizer is not None:
            del self._optimizer
            self._optimizer = None
        self._initialized = False

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Initialize torch.distributed if not already done."""
        import os

        import torch
        import torch.distributed as dist

        if dist.is_initialized():
            return

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if world_size <= 1:
            logger.info("Single-process mode, skipping dist init")
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            elif hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.set_device(0)
            return

        backend = "nccl"
        if hasattr(torch, "npu") and torch.npu.is_available():
            backend = "hccl"

        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)

        logger.info(
            "Initializing dist: rank=%d, world=%d, backend=%s",
            rank,
            world_size,
            backend,
        )
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            torch.npu.set_device(local_rank)

    def _load_model(self, model_path: str, dtype):
        """Load a HuggingFace causal LM."""
        from transformers import AutoModelForCausalLM

        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
        except (ImportError, ValueError):
            logger.info("flash_attention_2 not available, using default attention")
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        return model

    def _wrap_fsdp(self) -> None:
        """Wrap model with FSDP2 fully_shard."""
        import torch.distributed as dist

        if not dist.is_initialized():
            logger.warning(
                "torch.distributed not initialized — skipping FSDP wrapping. "
                "Model will run on a single device."
            )
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            return

        from torch.distributed.fsdp import fully_shard

        for module in self._model.modules():
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.numel() > 1_000_000:
                    fully_shard(module)

        fully_shard(self._model)
        logger.info("Model wrapped with FSDP2 fully_shard")

    def _create_scheduler(self):
        """Create cosine LR scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        warmup = self._config.warmup_steps
        max_steps = max(self._config.max_steps, 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            import math

            progress = (step - warmup) / max(max_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self._optimizer, lr_lambda)

    def _create_loss(self):
        """Create the loss function based on config."""
        loss_kwargs = self._config.loss_config

        if self._config.loss_type == "grpo":
            from forge.rl.loss import GRPOLoss

            return GRPOLoss(**loss_kwargs)
        elif self._config.loss_type == "dapo":
            from forge.rl.loss import DAPOLoss

            return DAPOLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {self._config.loss_type!r}")

    @staticmethod
    def _to_tensor(data, device, dtype=None):
        """Convert data to tensor on device."""
        import torch

        if isinstance(data, torch.Tensor):
            t = data.to(device=device)
            return t.to(dtype=dtype) if dtype is not None else t
        return torch.tensor(data, device=device, dtype=dtype)
