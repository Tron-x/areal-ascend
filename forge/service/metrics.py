"""Service metrics collection and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.service.replica import ReplicaMetrics


@dataclass
class ServiceMetrics:
    """Aggregated metrics for the entire service."""

    replica_metrics: dict[int, ReplicaMetrics] = field(default_factory=dict)
    total_sessions: int = 0
    healthy_replicas: int = 0
    total_replicas: int = 0
    last_scale_event: float = 0.0

    def get_total_request_rate(self, window_seconds: float = 60.0) -> float:
        return sum(
            m.get_request_rate(window_seconds) for m in self.replica_metrics.values()
        )

    def get_avg_queue_depth(self, replicas: list) -> float:
        healthy = [r for r in replicas if r.healthy]
        if not healthy:
            return 0.0
        total = sum(r.request_queue.qsize() for r in healthy)
        return total / len(healthy)

    def get_avg_capacity_utilization(self, replicas: list) -> float:
        healthy = [r for r in replicas if r.healthy]
        if not healthy:
            return 0.0
        total = sum(r.capacity_utilization for r in healthy)
        return total / len(healthy)

    def get_sessions_per_replica(self) -> float:
        if self.total_replicas == 0:
            return 0.0
        return self.total_sessions / self.total_replicas
