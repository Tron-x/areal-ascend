"""HuggingFace Reward Model engine for neural reward scoring.

Loads a HuggingFace AutoModelForSequenceClassification (or custom RM)
and runs batched GPU inference to score (prompt, response) pairs.

Supports two model types:

1. **Classifier RM** (default): ``AutoModelForSequenceClassification``
   with a scalar head. Common for RLHF reward models (e.g. OpenAssistant,
   UltraRM, Skywork-Reward).

2. **Generative RM** (``mode="generative"``): Uses an LLM as judge --
   generates a score token (e.g. "4") given a rubric prompt.  Slower but
   more flexible.

Usage::

    engine = HFRewardModelEngine(
        model_path="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        device="cuda:0",
        max_batch_size=16,
    )
    engine.load()
    scores = engine.score_batch([
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "What is 2+2?", "response": "Fish"},
    ])
    # scores ≈ [0.85, -0.12]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HFRewardModelEngine:
    """HuggingFace-based neural reward model engine.

    Satisfies the ``RewardModelEngine`` protocol from
    ``forge.core.protocols``.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_batch_size: int = 16,
        max_length: int = 2048,
        chat_template: str | None = None,
        score_index: int = 0,
        torch_compile: bool = False,
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path.
            device: Target device (``"auto"``, ``"cuda:0"``, ``"npu:0"``, etc.).
            dtype: Model dtype (``"float16"``, ``"bfloat16"``, ``"float32"``).
            max_batch_size: Max items per forward pass.
            max_length: Max input token length (truncate longer sequences).
            chat_template: Optional chat template for formatting. If None,
                uses the tokenizer's default or a simple concat.
            score_index: Which logit index to use as reward (for multi-head RMs).
            torch_compile: Whether to torch.compile the model for speed.
        """
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.chat_template = chat_template
        self.score_index = score_index
        self.torch_compile = torch_compile

        self._model = None
        self._tokenizer = None
        self._call_count = 0
        self._total_time = 0.0

    def load(self) -> dict:
        """Load model and tokenizer to device."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.dtype_str, torch.bfloat16)

        logger.info(
            f"Loading reward model: {self.model_path} "
            f"(device={self.device}, dtype={self.dtype_str})"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device if self.device != "auto" else "auto",
            trust_remote_code=True,
        )
        self._model.eval()

        if self.torch_compile:
            import torch

            self._model = torch.compile(self._model)

        actual_device = next(self._model.parameters()).device
        num_params = sum(p.numel() for p in self._model.parameters())
        num_labels = getattr(self._model.config, "num_labels", 1)

        logger.info(
            f"Reward model loaded: {num_params / 1e6:.1f}M params, "
            f"{num_labels} label(s), device={actual_device}"
        )
        return {
            "model_path": self.model_path,
            "device": str(actual_device),
            "num_params": num_params,
            "num_labels": num_labels,
            "status": "ready",
        }

    def _format_input(self, prompt: str, response: str) -> str:
        """Format (prompt, response) into a single string for the tokenizer."""
        if self.chat_template:
            return self.chat_template.replace("{prompt}", prompt).replace(
                "{response}", response
            )

        if self._tokenizer and self._tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                pass

        return f"User: {prompt}\nAssistant: {response}"

    def score(self, prompt: str, response: str) -> float:
        """Score a single (prompt, response) pair."""
        return self.score_batch([{"prompt": prompt, "response": response}])[0]

    def score_batch(self, items: list[dict[str, str]]) -> list[float]:
        """Score a batch of (prompt, response) pairs."""
        import time

        import torch

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not items:
            return []

        t0 = time.monotonic()
        all_scores: list[float] = []

        for batch_start in range(0, len(items), self.max_batch_size):
            batch = items[batch_start : batch_start + self.max_batch_size]
            texts = [
                self._format_input(item["prompt"], item["response"])
                for item in batch
            ]

            encodings = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            encodings = {k: v.to(device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self._model(**encodings)
                logits = outputs.logits

            if logits.dim() == 1:
                scores = logits.float().cpu().tolist()
            elif logits.shape[-1] == 1:
                scores = logits.squeeze(-1).float().cpu().tolist()
            else:
                scores = logits[:, self.score_index].float().cpu().tolist()

            all_scores.extend(scores)

        elapsed = time.monotonic() - t0
        self._call_count += len(items)
        self._total_time += elapsed

        return all_scores

    def get_stats(self) -> dict[str, Any]:
        """Return inference statistics."""
        avg = (self._total_time / self._call_count) if self._call_count > 0 else 0
        return {
            "model_path": self.model_path,
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time_per_item": avg,
        }

    def shutdown(self) -> None:
        """Free model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Reward model engine shut down")
