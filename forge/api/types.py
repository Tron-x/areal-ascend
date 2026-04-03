"""Core data types for Forge.

All types are framework-agnostic -- no Monarch, Ray, or vLLM imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SampleStatus(Enum):
    """Lifecycle state of a rollout sample."""

    PENDING = auto()
    COMPLETED = auto()
    TRUNCATED = auto()
    ABORTED = auto()
    FAILED = auto()


@dataclass
class SamplingParams:
    """Parameters for text generation."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    skip_special_tokens: bool = True
    return_logprobs: bool = True

    def copy(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            stop=list(self.stop),
            stop_token_ids=list(self.stop_token_ids),
            skip_special_tokens=self.skip_special_tokens,
            return_logprobs=self.return_logprobs,
        )


@dataclass
class Sample:
    """A single rollout sample flowing through the pipeline.

    Designed to be the universal data carrier between rollout functions,
    reward computation, and training -- framework-agnostic.
    """

    prompt: str
    label: str | None = None
    response: str = ""
    reward: float = 0.0
    status: SampleStatus = SampleStatus.PENDING

    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)

    response_length: int = 0
    task_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_response(
        self,
        response: str,
        reward: float,
        *,
        status: SampleStatus = SampleStatus.COMPLETED,
        response_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        loss_mask: list[int] | None = None,
    ) -> Sample:
        """Return a new Sample with the response and reward filled in."""
        return Sample(
            prompt=self.prompt,
            label=self.label,
            response=response,
            reward=reward,
            status=status,
            prompt_ids=self.prompt_ids,
            response_ids=response_ids or self.response_ids,
            logprobs=logprobs or self.logprobs,
            loss_mask=loss_mask or self.loss_mask,
            response_length=len(response_ids) if response_ids else len(response),
            task_data=self.task_data,
            metadata=self.metadata,
        )


@dataclass
class GenerateResult:
    """Result from a single generation call."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainBatch:
    """A batch of data ready for training.

    The ``tensors`` dict maps field names to tensors (framework-agnostic --
    can be torch.Tensor, numpy arrays, or lists depending on the adapter).
    """

    tensors: dict[str, Any] = field(default_factory=dict)
    policy_version: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of samples in the batch (inferred from first tensor)."""
        if not self.tensors:
            return 0
        first = next(iter(self.tensors.values()))
        if hasattr(first, "__len__"):
            return len(first)
        return 0


class Metrics(dict):
    """Training/rollout metrics -- a dict subclass with convenience methods."""

    def merge(self, other: Metrics) -> Metrics:
        """Merge another Metrics dict into this one (averages overlapping keys)."""
        merged = Metrics(self)
        for k, v in other.items():
            if k in merged and isinstance(merged[k], (int, float)):
                merged[k] = (merged[k] + v) / 2.0
            else:
                merged[k] = v
        return merged

    def prefixed(self, prefix: str) -> Metrics:
        """Return a copy with all keys prefixed."""
        return Metrics({f"{prefix}/{k}": v for k, v in self.items()})
