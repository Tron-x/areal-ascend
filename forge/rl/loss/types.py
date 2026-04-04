"""Loss types -- lightweight, no pydantic dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from forge.observability.metrics import Metric

AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
RatioType = Literal["token", "sequence"]
KLType = Literal["k1", "k2", "k3"]

CROSS_ENTROPY_IGNORE_IDX = -100


@dataclass
class LossOutput:
    """Output from all loss functions.

    Attributes:
        loss: Scalar loss tensor for backpropagation.
        metrics: List of Metric observations for logging.
    """

    loss: torch.Tensor
    metrics: list[Metric] = field(default_factory=list)
