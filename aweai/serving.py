"""Framework-neutral model-serving primitives for AWEAI."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import monotonic
from typing import Dict, Iterable, List

@dataclass(frozen=True)
class Replica:
    name: str
    capacity_rps: float
    healthy: bool = True

@dataclass
class ServingPool:
    replicas: List[Replica] = field(default_factory=list)
    requests: int = 0
    errors: int = 0
    latency_ms_total: float = 0.0

    def healthy(self) -> List[Replica]:
        return [r for r in self.replicas if r.healthy and r.capacity_rps > 0]

    def choose(self) -> Replica:
        candidates = self.healthy()
        if not candidates:
            raise RuntimeError("no healthy serving replica")
        return min(candidates, key=lambda r: r.name)

    def observe(self, latency_ms: float, error: bool = False) -> None:
        self.requests += 1
        self.latency_ms_total += max(0.0, latency_ms)
        self.errors += int(error)

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.requests if self.requests else 0.0

class Autoscaler:
    def __init__(self, min_replicas: int = 1, max_replicas: int = 16, target_utilization: float = .70):
        if min_replicas < 1 or max_replicas < min_replicas:
            raise ValueError("invalid replica bounds")
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_utilization = target_utilization

    def desired(self, replicas: int, observed_rps: float, capacity_per_replica: float) -> int:
        if capacity_per_replica <= 0:
            raise ValueError("capacity_per_replica must be positive")
        raw = int((observed_rps / (capacity_per_replica * self.target_utilization)) + .999999)
        return max(self.min_replicas, min(self.max_replicas, raw))


def serving_health(pool: ServingPool, max_error_rate: float = .02, max_latency_ms: float = 1000.0) -> bool:
    return bool(pool.healthy()) and pool.error_rate <= max_error_rate and pool.avg_latency_ms <= max_latency_ms
