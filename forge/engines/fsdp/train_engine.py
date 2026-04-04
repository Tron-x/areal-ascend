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
        """Load model, wrap with FSDP2, create optimizer and scheduler."""
        import torch
        import torch.distributed as dist

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
            rank, cfg.max_steps,
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
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
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
        """
        if not self._initialized:
            raise RuntimeError("FSDPTrainEngine not initialized")
        return {k: v.cpu() for k, v in self._model.state_dict().items()}

    def get_metadata(self) -> dict:
        return {
            "max_steps": self._config.max_steps,
            "current_step": self._current_step,
            "model_path": self._config.model_path,
            "loss_type": self._config.loss_type,
            "dtype": self._config.dtype,
        }

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

    def _load_model(self, model_path: str, dtype):
        """Load a HuggingFace causal LM."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
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
