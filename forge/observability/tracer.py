"""Performance tracer for timing code sections.

Lightweight alternative to TorchForge's ``perf_tracker.Tracer``.
Supports CPU timing with optional step labels for breakdown analysis.

Usage::

    t = Tracer("train_step")
    t.start()
    ... forward + backward ...
    t.step("forward_backward")
    ... optimizer ...
    t.step("optimizer")
    t.stop()  # logs full breakdown
"""

from __future__ import annotations

import logging
import time

from forge.observability.metrics import Reduce, record_metric

logger = logging.getLogger(__name__)


class Tracer:
    """Lightweight CPU timer with step-based breakdown.

    Args:
        name: Name prefix for metrics (e.g. ``"train_step"``).
        log_to_metrics: If True, record timings via ``record_metric``.
        log_to_logger: If True, also log via Python logger.
    """

    def __init__(
        self,
        name: str,
        log_to_metrics: bool = True,
        log_to_logger: bool = False,
    ):
        self.name = name
        self.log_to_metrics = log_to_metrics
        self.log_to_logger = log_to_logger
        self._start_time: float = 0.0
        self._step_time: float = 0.0
        self._steps: list[tuple[str, float]] = []

    def start(self) -> Tracer:
        self._start_time = time.monotonic()
        self._step_time = self._start_time
        self._steps = []
        return self

    def step(self, label: str) -> float:
        """Record a named sub-interval since the last step/start."""
        now = time.monotonic()
        elapsed = now - self._step_time
        self._steps.append((label, elapsed))
        self._step_time = now

        if self.log_to_metrics:
            record_metric(f"{self.name}/{label}", elapsed, Reduce.MEAN)
        return elapsed

    def stop(self) -> float:
        """Stop the tracer. Returns total elapsed time."""
        total = time.monotonic() - self._start_time

        if self.log_to_metrics:
            record_metric(f"{self.name}/total", total, Reduce.MEAN)

        if self.log_to_logger and self._steps:
            breakdown = ", ".join(f"{label}={dt:.3f}s" for label, dt in self._steps)
            logger.info(f"[{self.name}] total={total:.3f}s ({breakdown})")

        return total

    def __enter__(self) -> Tracer:
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()
