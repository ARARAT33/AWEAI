"""Deterministic production orchestration primitives for AWEAI.

Framework-neutral: no cloud provider or model vendor is required. The module
provides planning, dependency validation, resource-aware placement, retries,
and health-aware rollout decisions that higher-level adapters can execute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ResourcePool:
    name: str
    cpu: float
    memory_gb: float
    accelerator: float = 0.0


@dataclass(frozen=True)
class Workload:
    name: str
    cpu: float = 1.0
    memory_gb: float = 1.0
    accelerator: float = 0.0
    depends_on: Tuple[str, ...] = ()
    priority: int = 0
    max_retries: int = 2


@dataclass(frozen=True)
class Placement:
    workload: str
    pool: str
    retry_budget: int


class OrchestrationError(ValueError):
    """Raised when a workload graph cannot be planned safely."""


class AWEAIOrchestrator:
    """Create deterministic, resource-aware execution plans."""

    def __init__(self, pools: Iterable[ResourcePool]):
        self.pools = tuple(pools)
        if not self.pools:
            raise OrchestrationError("at least one resource pool is required")

    @staticmethod
    def _validate_graph(workloads: Sequence[Workload]) -> Dict[str, Workload]:
        by_name = {w.name: w for w in workloads}
        if len(by_name) != len(workloads):
            raise OrchestrationError("duplicate workload name")
        for w in workloads:
            missing = [d for d in w.depends_on if d not in by_name]
            if missing:
                raise OrchestrationError(f"{w.name}: missing dependencies: {missing}")
        visiting, visited = set(), set()

        def visit(name: str) -> None:
            if name in visiting:
                raise OrchestrationError("dependency cycle detected")
            if name in visited:
                return
            visiting.add(name)
            for dep in by_name[name].depends_on:
                visit(dep)
            visiting.remove(name)
            visited.add(name)

        for name in by_name:
            visit(name)
        return by_name

    @staticmethod
    def _fits(w: Workload, p: ResourcePool) -> bool:
        return w.cpu <= p.cpu and w.memory_gb <= p.memory_gb and w.accelerator <= p.accelerator

    def plan(self, workloads: Sequence[Workload]) -> List[Placement]:
        by_name = self._validate_graph(workloads)
        remaining = set(by_name)
        done = set()
        result: List[Placement] = []
        while remaining:
            ready = sorted(
                (by_name[n] for n in remaining if set(by_name[n].depends_on) <= done),
                key=lambda w: (-w.priority, w.name),
            )
            if not ready:
                raise OrchestrationError("unable to resolve execution graph")
            for w in ready:
                candidates = [p for p in self.pools if self._fits(w, p)]
                if not candidates:
                    raise OrchestrationError(f"no resource pool can host {w.name}")
                pool = min(candidates, key=lambda p: (p.accelerator < w.accelerator, p.name))
                result.append(Placement(w.name, pool.name, max(0, w.max_retries)))
                done.add(w.name)
                remaining.remove(w.name)
        return result

    def fingerprint(self, workloads: Sequence[Workload]) -> str:
        plan = self.plan(workloads)
        payload = "|".join(f"{x.workload}:{x.pool}:{x.retry_budget}" for x in plan)
        return sha256(payload.encode()).hexdigest()


@dataclass
class RolloutState:
    desired: int
    healthy: int = 0
    error_rate: float = 0.0
    traffic_percent: int = 0
    history: List[int] = field(default_factory=list)


class CanaryPolicy:
    """Fail-closed canary progression policy."""

    def __init__(self, error_rate_limit: float = 0.02, min_healthy_ratio: float = 0.99):
        self.error_rate_limit = error_rate_limit
        self.min_healthy_ratio = min_healthy_ratio

    def decide(self, state: RolloutState, next_traffic: int) -> str:
        ratio = state.healthy / state.desired if state.desired else 0.0
        if ratio < self.min_healthy_ratio or state.error_rate > self.error_rate_limit:
            return "rollback"
        if next_traffic <= state.traffic_percent:
            return "hold"
        return "promote"
