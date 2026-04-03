"""Metrics collection with accumulator-based reduction.

Key design (from TorchForge):
- ``record_metric(key, value, Reduce.MEAN)`` → thread-local accumulator
- ``flush_metrics(step)`` → reduce all accumulators, return dict, reset
- ``reduce_metrics_states(states)`` → merge accumulator states across ranks
  (more precise than averaging pre-reduced values)

No Monarch dependency.  Works in any Python process.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Reduce(Enum):
    """Reduction strategy for metric accumulation."""

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    STD = "std"

    @property
    def accumulator_class(self) -> type[MetricAccumulator]:
        return _ACCUMULATORS[self]


@dataclass
class Metric:
    """A single metric observation."""

    key: str
    value: Any
    reduction: Reduce = Reduce.MEAN
    timestamp: float | None = None


# ======================================================================
# Accumulators
# ======================================================================


class MetricAccumulator(ABC):
    """Accumulates values and produces a reduced scalar."""

    def __init__(self, reduction: Reduce):
        self.reduction_type = reduction

    @abstractmethod
    def append(self, value: Any) -> None: ...

    @abstractmethod
    def get_value(self) -> float: ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...


class MeanAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce):
        super().__init__(reduction)
        self.sum = 0.0
        self.count = 0

    def append(self, value: Any) -> None:
        self.sum += float(value.item() if hasattr(value, "item") else value)
        self.count += 1

    def get_value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def get_state(self) -> dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "count": self.count,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_count = sum(s["count"] for s in states)
        return total_sum / total_count if total_count > 0 else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


class SumAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce):
        super().__init__(reduction)
        self.total = 0.0

    def append(self, value: Any) -> None:
        self.total += float(value.item() if hasattr(value, "item") else value)

    def get_value(self) -> float:
        return self.total

    def get_state(self) -> dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "total": self.total}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return sum(s["total"] for s in states)

    def reset(self) -> None:
        self.total = 0.0


class MaxAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce):
        super().__init__(reduction)
        self.max_val = float("-inf")

    def append(self, value: Any) -> None:
        self.max_val = max(
            self.max_val, float(value.item() if hasattr(value, "item") else value)
        )

    def get_value(self) -> float:
        return self.max_val

    def get_state(self) -> dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "max_val": self.max_val}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return max(s["max_val"] for s in states)

    def reset(self) -> None:
        self.max_val = float("-inf")


class MinAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce):
        super().__init__(reduction)
        self.min_val = float("inf")

    def append(self, value: Any) -> None:
        self.min_val = min(
            self.min_val, float(value.item() if hasattr(value, "item") else value)
        )

    def get_value(self) -> float:
        return self.min_val

    def get_state(self) -> dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "min_val": self.min_val}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return min(s["min_val"] for s in states)

    def reset(self) -> None:
        self.min_val = float("inf")


class StdAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce):
        super().__init__(reduction)
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.sum += v
        self.sum_sq += v * v
        self.count += 1

    def get_value(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def get_state(self) -> dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "sum_sq": self.sum_sq,
            "count": self.count,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_sum_sq = sum(s["sum_sq"] for s in states)
        total_count = sum(s["count"] for s in states)
        if total_count < 2:
            return 0.0
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def reset(self) -> None:
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0


_ACCUMULATORS: dict[Reduce, type[MetricAccumulator]] = {
    Reduce.MEAN: MeanAccumulator,
    Reduce.SUM: SumAccumulator,
    Reduce.MAX: MaxAccumulator,
    Reduce.MIN: MinAccumulator,
    Reduce.STD: StdAccumulator,
}


# ======================================================================
# Collector (thread-safe singleton)
# ======================================================================


class MetricCollector:
    """Thread-safe metric accumulator.

    Unlike TorchForge's per-rank singleton (requires Monarch), this
    uses a thread-local default instance accessible via ``get_collector()``.
    """

    def __init__(self) -> None:
        self.accumulators: dict[str, MetricAccumulator] = {}
        self.backends: list[LoggerBackend] = []
        self._lock = threading.Lock()

    def push(self, metric: Metric) -> None:
        with self._lock:
            key = metric.key
            if key not in self.accumulators:
                self.accumulators[key] = metric.reduction.accumulator_class(
                    metric.reduction
                )
            self.accumulators[key].append(metric.value)

    def flush(self, step: int = 0) -> dict[str, float]:
        """Reduce all accumulators, log to backends, reset, return dict."""
        with self._lock:
            results = {}
            states = {}
            for key, acc in self.accumulators.items():
                results[key] = acc.get_value()
                states[key] = acc.get_state()
                acc.reset()

        for backend in self.backends:
            backend.log_step(results, step)

        return results

    def get_states(self) -> dict[str, dict[str, Any]]:
        """Return serializable accumulator states for cross-rank merge."""
        with self._lock:
            return {key: acc.get_state() for key, acc in self.accumulators.items()}

    def add_backend(self, backend: LoggerBackend) -> None:
        self.backends.append(backend)


_default_collector: MetricCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> MetricCollector:
    """Get or create the default MetricCollector."""
    global _default_collector
    with _collector_lock:
        if _default_collector is None:
            _default_collector = MetricCollector()
        return _default_collector


def record_metric(
    key: str,
    value: Any,
    reduction: Reduce = Reduce.MEAN,
    timestamp: float | None = None,
) -> None:
    """Record a metric value. Thread-safe."""
    if os.getenv("FORGE_DISABLE_METRICS", "false").lower() == "true":
        return
    metric = Metric(
        key=key, value=value, reduction=reduction, timestamp=timestamp or time.time()
    )
    get_collector().push(metric)


def flush_metrics(step: int = 0) -> dict[str, float]:
    """Flush all accumulated metrics, log to backends, return reduced values."""
    return get_collector().flush(step)


def reduce_metrics_states(states: list[dict[str, dict[str, Any]]]) -> dict[str, float]:
    """Merge accumulator states from multiple ranks into reduced values.

    More precise than averaging pre-reduced floats because it merges
    the raw accumulator state (sum/count for MEAN, etc.).
    """
    if not states:
        return {}

    all_keys = set(k for state in states for k in state)
    reduced = {}

    for key in all_keys:
        metric_states = [s[key] for s in states if key in s]
        if not metric_states:
            continue
        reduction_type = Reduce(metric_states[0]["reduction_type"])
        acc_cls = reduction_type.accumulator_class
        reduced[key] = acc_cls.get_reduced_value_from_states(metric_states)

    return reduced


# ======================================================================
# Logger backends
# ======================================================================


class LoggerBackend(ABC):
    """Abstract backend for metric output (console, wandb, file, etc.)."""

    @abstractmethod
    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """Log a batch of reduced metrics for a training step."""
        ...


class ConsoleBackend(LoggerBackend):
    """Print metrics to stdout."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        header = f"[{self.prefix}] " if self.prefix else ""
        lines = [f"  {k}: {v:.4f}" for k, v in sorted(metrics.items())]
        print(f"{header}Step {step}:\n" + "\n".join(lines))
