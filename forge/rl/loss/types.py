"""Loss types -- lightweight, no pydantic dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
RatioType = Literal["token", "sequence"]
KLType = Literal["k1", "k2", "k3"]

CROSS_ENTROPY_IGNORE_IDX = -100


@dataclass
class Metric:
    """A single metric observation for logging."""

    key: str
    value: float | torch.Tensor
    reduction: str = "mean"


@dataclass
class LossOutput:
    """Output from all loss functions.

    Attributes:
        loss: Scalar loss tensor for backpropagation.
        metrics: List of Metric observations for logging.
    """

    loss: torch.Tensor
    metrics: list[Metric] = field(default_factory=list)
