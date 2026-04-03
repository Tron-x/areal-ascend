"""Forge observability -- metrics collection, timing, and logging.

Provides a lightweight, Forge-native metrics system independent of any
training framework's logging (AReaL stats_tracker, etc.).

Core API::

    from forge.observability import record_metric, Reduce, flush_metrics

    record_metric("loss/pg", 0.42, Reduce.MEAN)
    record_metric("rollout/reward", 0.85, Reduce.MEAN)
    record_metric("rollout/episodes", 1, Reduce.SUM)

    metrics = flush_metrics(step=10)  # returns reduced {key: value}

Timing::

    from forge.observability import Tracer

    t = Tracer("train_step")
    t.start()
    ... do work ...
    t.step("forward")
    ... more work ...
    t.stop()  # logs timing breakdown
"""

from forge.observability.metrics import (
    ConsoleBackend,
    LoggerBackend,
    MetricCollector,
    Reduce,
    flush_metrics,
    get_collector,
    record_metric,
    reduce_metrics_states,
)
from forge.observability.tracer import Tracer

__all__ = [
    "ConsoleBackend",
    "LoggerBackend",
    "MetricCollector",
    "Reduce",
    "Tracer",
    "flush_metrics",
    "get_collector",
    "record_metric",
    "reduce_metrics_states",
]
