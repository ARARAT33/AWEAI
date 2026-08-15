"""AWEAI-native deterministic intelligence primitives.

These are engineering algorithms, not chat or agent features.  They provide
stable building blocks for routing, planning, provenance, optimisation and
consistency checks inside AWEAI.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _hash(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CapabilityScore:
    name: str
    score: float
    reasons: Tuple[str, ...] = ()


class AWEIAdaptiveRouter:
    """Rank capabilities using utility, reliability, risk, cost and feedback."""

    def rank(
        self,
        candidates: Mapping[str, Mapping[str, float]],
        required: Mapping[str, float],
        history: Mapping[str, float] = None,
    ) -> List[CapabilityScore]:
        history = history or {}
        out = []
        for name, meta in candidates.items():
            utility = sum(float(required.get(k, 0.0)) * float(meta.get(k, 0.0)) for k in required)
            risk = float(meta.get("risk", 0.0))
            cost = float(meta.get("cost", 0.0))
            reliability = float(meta.get("reliability", 1.0))
            score = utility * (0.5 + 0.5 * reliability)
            score += float(history.get(name, 0.0))
            score -= 0.35 * risk + 0.15 * cost
            out.append(CapabilityScore(
                name,
                score,
                (f"utility={utility:.4f}", f"reliability={reliability:.3f}"),
            ))
        return sorted(out, key=lambda item: item.score, reverse=True)


class AWEAIWorkloadPlanner:
    """Convert a dependency DAG into deterministic parallel execution waves."""

    def waves(self, nodes: Mapping[str, Sequence[str]]) -> List[List[str]]:
        deps = {key: set(value) for key, value in nodes.items()}
        waves: List[List[str]] = []
        while deps:
            ready = sorted(key for key, value in deps.items() if not value)
            if not ready:
                raise ValueError("cyclic workload graph")
            waves.append(ready)
            for key in ready:
                deps.pop(key)
            for value in deps.values():
                value.difference_update(ready)
        return waves


class AWEAIProvenanceChain:
    """Create a tamper-evident hash chain for datasets, models and results."""

    def __init__(self) -> None:
        self._last = "0" * 64

    def append(self, event: Mapping) -> str:
        record = {"previous": self._last, "event": dict(event)}
        self._last = _hash(record)
        return self._last

    @property
    def head(self) -> str:
        return self._last


class AWEAIFrontierOptimizer:
    """Small deterministic derivative-free search for engineering configurations."""

    def search(
        self,
        dimensions: Mapping[str, Sequence[float]],
        objective,
        rounds: int = 3,
    ) -> Tuple[Dict[str, float], float]:
        if any(not values for values in dimensions.values()):
            raise ValueError("optimization dimensions cannot be empty")
        current = {key: float(values[len(values) // 2]) for key, values in dimensions.items()}
        best = float(objective(current))
        for _ in range(max(1, int(rounds))):
            for key, values in dimensions.items():
                for value in values:
                    trial = dict(current)
                    trial[key] = float(value)
                    score = float(objective(trial))
                    if score > best:
                        current, best = trial, score
        return current, best


class AWEAIConsistencyEngine:
    """Measure deterministic agreement between repeated numeric observations."""

    def score(self, observations: Iterable[Mapping[str, float]]) -> float:
        rows = list(observations)
        if not rows:
            return 1.0
        keys = set().union(*(row.keys() for row in rows))
        penalties = 0.0
        for key in keys:
            values = [float(row[key]) for row in rows if key in row]
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                penalties += variance / (1.0 + abs(mean))
        return 1.0 / (1.0 + penalties)


# Explicit public compatibility aliases.  Keep these names stable so external
# callers and CI can import the primitives directly across AWEAI releases.
WorkloadPlanner = AWEAIWorkloadPlanner
AdaptiveRouter = AWEIAdaptiveRouter
ProvenanceChain = AWEAIProvenanceChain
FrontierOptimizer = AWEAIFrontierOptimizer
ConsistencyEngine = AWEAIConsistencyEngine

__all__ = [
    "CapabilityScore",
    "AWEIAdaptiveRouter",
    "AWEAIWorkloadPlanner",
    "AWEAIProvenanceChain",
    "AWEAIFrontierOptimizer",
    "AWEAIConsistencyEngine",
    "WorkloadPlanner",
    "AdaptiveRouter",
    "ProvenanceChain",
    "FrontierOptimizer",
    "ConsistencyEngine",
]
