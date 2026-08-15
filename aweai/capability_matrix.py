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

    def __post_init__(self) -> None:
        for field in ("implementation", "tests", "production"):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1: {value!r}")
        if not self.name.strip():
            raise ValueError("capability name must not be empty")
        if not self.domain.strip():
            raise ValueError("capability domain must not be empty")

    @property
    def score(self) -> float:
        return round((self.implementation + self.tests + self.production) / 3.0, 2)

    @property
    def status(self) -> str:
        if self.score >= 0.80:
            return "production"
        if self.score >= 0.65:
            return "readying"
        if self.score >= 0.50:
            return "developing"
        return "early"


# Keep this matrix conservative: values reflect the current repository's
# implemented/tested surface, including the distributed planner and release
# capability work. This is intentionally not a marketing score.
DEFAULT_CAPABILITIES: tuple[Capability, ...] = (
    Capability("model-engineering", "models", .60, .60, .50),
    Capability("data-engineering", "data", .50, .45, .40),
    Capability("research", "research", .45, .40, .35),
    Capability("multimodal", "models", .40, .35, .30),
    Capability("inference-routing", "runtime", .60, .55, .50),
    Capability("evaluation-benchmarks", "quality", .60, .60, .55),
    Capability("artifact-lineage", "quality", .65, .60, .60),
    Capability("release-gates", "release", .70, .70, .65),
    Capability("slo-canary", "operations", .55, .50, .45),
    Capability("cost-governance", "operations", .55, .45, .45),
    Capability("security-governance", "security", .45, .40, .35),
    Capability("infrastructure", "platform", .45, .35, .30),
    Capability("software-engineering", "engineering", .65, .65, .60),
    Capability("knowledge-rag", "knowledge", .45, .40, .35),
    Capability("company-platform", "platform", .55, .55, .50),
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

    def weakest(self, limit: int = 5) -> List[dict]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        ordered = sorted(self.capabilities, key=lambda c: (c.score, c.name))
        return [
            {"name": c.name, "domain": c.domain, "score": c.score, "status": c.status}
            for c in ordered[:limit]
        ]

    def gate_diagnostics(self, minimum: float = .55) -> dict:
        if not 0.0 <= minimum <= 1.0:
            raise ValueError("minimum must be between 0 and 1")
        floor = round(minimum * 0.75, 2)
        below_floor = [c.name for c in self.capabilities if c.score < floor]
        return {
            "minimum": minimum,
            "capability_floor": floor,
            "overall": self.score,
            "overall_pass": self.score >= minimum,
            "below_floor": below_floor,
            "passed": self.score >= minimum and not below_floor,
        }

    def report(self) -> dict:
        return {
            "overall": self.score,
            "overall_percent": round(self.score * 100, 1),
            "status": "production" if self.score >= .80 else "readying" if self.score >= .65 else "developing" if self.score >= .50 else "early",
            "capabilities": [
                asdict(c) | {"score": c.score, "status": c.status}
                for c in self.capabilities
            ],
            "domains": self.domains(),
            "weakest": self.weakest(),
        }

    def gate(self, minimum: float = .55) -> bool:
        return self.gate_diagnostics(minimum)["passed"]
