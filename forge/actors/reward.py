"""RewardActor — unified reward computation for RL training.

Supports three reward modes:

1. **Rule-based** (default): Loads a Python callable via import path.
   CPU-only, no GPU needed. Uses ``RewardBackend`` protocol.

2. **Model-based**: Loads a neural reward model (HuggingFace
   ``AutoModelForSequenceClassification``) for GPU inference.
   Uses ``RewardModelEngine`` protocol.

3. **Hybrid**: Combines rule-based and model-based scores with
   configurable weights.

Deploy as a service for parallel reward evaluation::

    # Rule-based (CPU)
    reward = await RewardActor.options(procs=1).as_actor()
    await reward.setup.call(reward_fn_path="areal.reward.gsm8k.gsm8k_reward_fn")

    # Model-based (GPU)
    reward = await RewardActor.options(procs=1, with_gpus=True).as_actor()
    await reward.setup_model.call(model_path="Skywork/Skywork-Reward-8B")

    # Hybrid
    reward = await RewardActor.options(procs=1, with_gpus=True).as_actor()
    await reward.setup.call(reward_fn_path="areal.reward.gsm8k.gsm8k_reward_fn")
    await reward.setup_model.call(model_path="Skywork/Skywork-Reward-8B")
    await reward.set_mode.call(mode="hybrid", rule_weight=0.5, model_weight=0.5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

if TYPE_CHECKING:
    from forge.core.protocols import RewardBackend, RewardModelEngine

logger = logging.getLogger(__name__)


class RewardActor(ForgeActor):
    """Monarch Actor for reward computation (rule-based, model-based, or hybrid).

    Attributes:
        procs: Default 1 process.
        with_gpus: False for rule-only, True when using a reward model.
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        backend: RewardBackend | None = None,
        model_engine: RewardModelEngine | None = None,
    ):
        self._backend = backend
        self._model_engine = model_engine
        self._mode = "rule"
        self._rule_weight = 1.0
        self._model_weight = 0.0

    def _ensure_backend(self) -> RewardBackend:
        if self._backend is None:
            from forge.engines.areal.reward_backend import AReaLRewardBackend

            self._backend = AReaLRewardBackend()
        return self._backend

    # ------------------------------------------------------------------
    # Setup endpoints
    # ------------------------------------------------------------------

    @endpoint
    def setup(self, reward_fn_path: str = "") -> dict:
        """Load a rule-based reward function via import path."""
        return self._ensure_backend().setup(reward_fn_path)

    @endpoint
    def setup_model(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_batch_size: int = 16,
        max_length: int = 2048,
        chat_template: str | None = None,
        torch_compile: bool = False,
    ) -> dict:
        """Load a neural reward model for GPU inference."""
        from forge.engines.reward_model import HFRewardModelEngine

        self._model_engine = HFRewardModelEngine(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_length=max_length,
            chat_template=chat_template,
            torch_compile=torch_compile,
        )
        result = self._model_engine.load()

        if self._mode == "rule" and self._backend is None:
            self._mode = "model"
            self._rule_weight = 0.0
            self._model_weight = 1.0
        elif self._mode == "rule":
            self._mode = "hybrid"
            self._rule_weight = 0.5
            self._model_weight = 0.5

        logger.info(f"Reward mode set to '{self._mode}' after model load")
        return result

    @endpoint
    def set_mode(
        self,
        mode: str = "rule",
        rule_weight: float = 1.0,
        model_weight: float = 0.0,
    ) -> dict:
        """Set reward computation mode and weights.

        Args:
            mode: ``"rule"``, ``"model"``, or ``"hybrid"``.
            rule_weight: Weight for rule-based score in hybrid mode.
            model_weight: Weight for model-based score in hybrid mode.
        """
        if mode not in ("rule", "model", "hybrid"):
            raise ValueError(
                f"Invalid mode: {mode!r}. Use 'rule', 'model', or 'hybrid'."
            )
        if mode in ("model", "hybrid") and self._model_engine is None:
            raise RuntimeError(
                f"Cannot set mode={mode!r}: no model loaded. Call setup_model() first."
            )
        self._mode = mode
        self._rule_weight = rule_weight
        self._model_weight = model_weight
        logger.info(
            f"Reward mode: {mode} "
            f"(rule_weight={rule_weight}, model_weight={model_weight})"
        )
        return {"mode": mode, "rule_weight": rule_weight, "model_weight": model_weight}

    # ------------------------------------------------------------------
    # Scoring endpoints
    # ------------------------------------------------------------------

    @endpoint
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float:
        """Compute reward for a single prompt-completion pair."""
        return self._compute_single(
            prompt=prompt,
            completion=completion,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            task_data=task_data,
        )

    @endpoint
    def compute_rewards_batch(self, items: list[dict]) -> list[float]:
        """Compute rewards for a batch of items."""
        if self._mode == "model":
            return self._score_model_batch(items)
        if self._mode == "hybrid":
            return self._score_hybrid_batch(items)
        return self._score_rule_batch(items)

    @endpoint
    def score_model(self, prompt: str, response: str) -> float:
        """Score using ONLY the neural reward model (bypass mode setting)."""
        if self._model_engine is None:
            raise RuntimeError("No reward model loaded. Call setup_model() first.")
        return self._model_engine.score(prompt, response)

    @endpoint
    def score_model_batch(self, items: list[dict[str, str]]) -> list[float]:
        """Batch score using ONLY the neural reward model."""
        if self._model_engine is None:
            raise RuntimeError("No reward model loaded. Call setup_model() first.")
        return self._model_engine.score_batch(items)

    @endpoint
    def get_stats(self) -> dict[str, Any]:
        """Return combined stats from rule backend and model engine."""
        stats: dict[str, Any] = {"mode": self._mode}
        if self._backend is not None:
            stats["rule"] = self._backend.get_stats()
        if self._model_engine is not None:
            stats["model"] = self._model_engine.get_stats()
        return stats

    @endpoint
    def get_mode(self) -> dict:
        """Return current reward mode and weights."""
        return {
            "mode": self._mode,
            "rule_weight": self._rule_weight,
            "model_weight": self._model_weight,
            "has_rule_backend": self._backend is not None,
            "has_model_engine": self._model_engine is not None,
        }

    @endpoint
    def shutdown_model(self) -> None:
        """Free the reward model GPU memory without shutting down the actor."""
        if self._model_engine is not None:
            self._model_engine.shutdown()
            self._model_engine = None
        if self._mode in ("model", "hybrid"):
            self._mode = "rule"
            self._rule_weight = 1.0
            self._model_weight = 0.0
        logger.info("Reward model unloaded, reverted to rule mode")

    # ------------------------------------------------------------------
    # Internal scoring logic
    # ------------------------------------------------------------------

    def _compute_single(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float:
        if self._mode == "rule":
            return self._ensure_backend().compute_reward(
                prompt=prompt,
                completion=completion,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                task_data=task_data,
            )
        if self._mode == "model":
            return self._model_engine.score(prompt, completion)

        rule_score = self._ensure_backend().compute_reward(
            prompt=prompt,
            completion=completion,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            task_data=task_data,
        )
        model_score = self._model_engine.score(prompt, completion)
        return self._rule_weight * rule_score + self._model_weight * model_score

    def _score_rule_batch(self, items: list[dict]) -> list[float]:
        return self._ensure_backend().compute_rewards_batch(items)

    def _score_model_batch(self, items: list[dict]) -> list[float]:
        if self._model_engine is None:
            raise RuntimeError("No reward model loaded.")
        batch = [
            {"prompt": it.get("prompt", ""), "response": it.get("completion", "")}
            for it in items
        ]
        return self._model_engine.score_batch(batch)

    def _score_hybrid_batch(self, items: list[dict]) -> list[float]:
        rule_scores = self._score_rule_batch(items)
        model_scores = self._score_model_batch(items)
        return [
            self._rule_weight * r + self._model_weight * m
            for r, m in zip(rule_scores, model_scores)
        ]
