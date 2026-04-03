"""Engine protocols: the central decoupling point for Forge.

Two levels of abstraction:

**Low-level** (user-facing, used inside ``@rollout_fn``):
- ``GenerateEngine``: per-prompt text generation.

**High-level** (pipeline-facing, used by ``run_pipeline``):
- ``RolloutStage``: produces a training batch per step.
- ``TrainStage``: consumes a training batch and drives weight sync.

Adapters implement these protocols so the pipeline layer has zero
framework coupling.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from forge.api.types import GenerateResult, SamplingParams

# ---------------------------------------------------------------------------
# Low-level protocol: used by @rollout_fn authors
# ---------------------------------------------------------------------------


@runtime_checkable
class GenerateEngine(Protocol):
    """Per-prompt text generation — what users call inside ``@rollout_fn``.

    Implementations: Monarch vLLM, SGLang HTTP, mock engine, etc.
    """

    async def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerateResult]:
        """Generate completions for a batch of prompts."""
        ...

    async def update_weights(self, version: int) -> None:
        """Hot-reload model weights after a training step."""
        ...

    async def shutdown(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# High-level protocols: used by run_pipeline()
# ---------------------------------------------------------------------------


@runtime_checkable
class RolloutStage(Protocol):
    """Produce one training batch per step.

    Default adapter implementation runs ``GenerateEngine`` + ``RewardFn``
    (or the full AReaL rollout workflow) and returns a dict batch ready
    for training.
    """

    async def produce_batch(self, step: int) -> dict[str, Any]:
        """Run rollout for *step* and return a training-ready batch dict."""
        ...


@runtime_checkable
class TrainStage(Protocol):
    """Consume a training batch, update the model, and sync weights.

    A single ``consume_batch`` call covers the full cycle:
    advantage computation → PPO update → weight sync → logging.
    """

    async def consume_batch(self, batch: dict[str, Any], step: int) -> dict[str, Any]:
        """Train on *batch* for *step*. Returns metrics dict."""
        ...

    async def get_info(self) -> dict[str, Any]:
        """Return pipeline metadata (max_steps, start_step, etc.)."""
        ...
