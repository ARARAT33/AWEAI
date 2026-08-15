"""AWEAI production operations primitives.

Model-agnostic, offline-capable building blocks for running AI products:
release manifests, SLO gates, canary rollouts, cost budgets, reproducibility,
and incident state.  These are engineering primitives, not chat or agent APIs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReleaseManifest:
    product: str
    version: str
    artifact_id: str
    environment: str = "production"
    config_digest: str = ""
    code_digest: str = ""

    def fingerprint(self) -> str:
        return stable_digest(self.__dict__)


@dataclass(frozen=True)
class SLOGate:
    availability_min: float = 0.999
    p95_latency_max_ms: float = 1000.0
    error_rate_max: float = 0.01

    def evaluate(self, availability: float, p95_latency_ms: float, error_rate: float) -> Tuple[bool, Tuple[str, ...]]:
        failures = []
        if availability < self.availability_min:
            failures.append("availability")
        if p95_latency_ms > self.p95_latency_max_ms:
            failures.append("latency")
        if error_rate > self.error_rate_max:
            failures.append("error_rate")
        return not failures, tuple(failures)


@dataclass(frozen=True)
class CanaryDecision:
    promote: bool
    traffic_percent: float
    reasons: Tuple[str, ...] = ()


class AWEAICanaryController:
    """Fail-closed progressive delivery decision engine."""

    def decide(self, *, baseline: SLOGate, canary_metrics: Mapping[str, float], traffic_percent: float = 5.0) -> CanaryDecision:
        if not 0.0 < traffic_percent <= 100.0:
            raise ValueError("traffic_percent must be in (0, 100]")
        ok, failures = baseline.evaluate(
            float(canary_metrics.get("availability", 0.0)),
            float(canary_metrics.get("p95_latency_ms", float("inf"))),
            float(canary_metrics.get("error_rate", 1.0)),
        )
        return CanaryDecision(ok, traffic_percent if ok else 0.0, failures)


@dataclass(frozen=True)
class CostBudget:
    max_daily_cost: float
    currency: str = "USD"

    def check(self, projected_daily_cost: float) -> bool:
        if projected_daily_cost < 0:
            raise ValueError("projected_daily_cost cannot be negative")
        return projected_daily_cost <= self.max_daily_cost


class AWEAICostOptimizer:
    """Deterministic selection of the cheapest eligible execution option."""

    def choose(self, options: Mapping[str, Mapping[str, float]], *, budget: float) -> Optional[str]:
        eligible = []
        for name, item in options.items():
            cost = float(item.get("cost", float("inf")))
            score = float(item.get("score", 0.0))
            if cost <= budget:
                eligible.append((cost, -score, name))
        return min(eligible)[2] if eligible else None


class AWEAIReproducibility:
    """Create a stable experiment fingerprint from code/config/data metadata."""

    @staticmethod
    def fingerprint(*, code: Any, config: Any, data: Any, environment: Any = None) -> str:
        return stable_digest({"code": code, "config": config, "data": data, "environment": environment})


class AWEAIIncidentState:
    """Small deterministic incident state machine for production automation."""

    _allowed = {
        "open": {"acknowledged", "resolved"},
        "acknowledged": {"mitigating", "resolved"},
        "mitigating": {"resolved", "open"},
        "resolved": set(),
    }

    def __init__(self) -> None:
        self.state = "open"

    def transition(self, target: str) -> str:
        if target not in self._allowed.get(self.state, set()):
            raise ValueError(f"invalid incident transition: {self.state} -> {target}")
        self.state = target
        return self.state


__all__ = [
    "stable_digest",
    "ReleaseManifest",
    "SLOGate",
    "CanaryDecision",
    "AWEAICanaryController",
    "CostBudget",
    "AWEAICostOptimizer",
    "AWEAIReproducibility",
    "AWEAIIncidentState",
]
