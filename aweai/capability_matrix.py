"""AWEAI capability coverage and production-readiness scoring.

This module intentionally measures implemented capabilities rather than making
claims about parity with any external company. Scores are deterministic and
machine-readable so CI/release tooling can track progress over time.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Capability:
    name: str
    domain: str
    implementation: float
    tests: float
    production: float

    @property
    def score(self) -> float:
        return round((self.implementation + self.tests + self.production) / 3.0, 2)


DEFAULT_CAPABILITIES: tuple[Capability, ...] = (
    Capability("model-engineering", "models", .55, .55, .45),
    Capability("data-engineering", "data", .50, .45, .40),
    Capability("research", "research", .45, .40, .35),
    Capability("multimodal", "models", .40, .35, .30),
    Capability("inference-routing", "runtime", .55, .50, .45),
    Capability("evaluation-benchmarks", "quality", .60, .60, .55),
    Capability("artifact-lineage", "quality", .65, .60, .60),
    Capability("release-gates", "release", .65, .65, .60),
    Capability("slo-canary", "operations", .55, .50, .45),
    Capability("cost-governance", "operations", .55, .45, .45),
    Capability("security-governance", "security", .45, .40, .35),
    Capability("infrastructure", "platform", .45, .35, .30),
    Capability("software-engineering", "engineering", .60, .60, .55),
    Capability("knowledge-rag", "knowledge", .45, .40, .35),
    Capability("company-platform", "platform", .50, .45, .40),
)


class CapabilityMatrix:
    """Deterministic capability matrix used by product/release automation."""

    def __init__(self, capabilities: Iterable[Capability] = DEFAULT_CAPABILITIES):
        self.capabilities: List[Capability] = list(capabilities)

    @property
    def score(self) -> float:
        if not self.capabilities:
            return 0.0
        return round(sum(c.score for c in self.capabilities) / len(self.capabilities), 2)

    def domains(self) -> Dict[str, float]:
        groups: Dict[str, List[float]] = {}
        for capability in self.capabilities:
            groups.setdefault(capability.domain, []).append(capability.score)
        return {k: round(sum(v) / len(v), 2) for k, v in sorted(groups.items())}

    def report(self) -> dict:
        return {
            "overall": self.score,
            "overall_percent": round(self.score * 100, 1),
            "capabilities": [asdict(c) | {"score": c.score} for c in self.capabilities],
            "domains": self.domains(),
        }

    def gate(self, minimum: float = .55) -> bool:
        return self.score >= minimum and all(c.score >= minimum * .75 for c in self.capabilities)
