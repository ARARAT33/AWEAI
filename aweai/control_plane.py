"""Deterministic production control-plane primitives for AWEAI.

These APIs are intentionally model/provider agnostic. They provide policy,
execution planning, health observations, and audit records without pretending
to replace external infrastructure.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Policy:
    name: str
    max_cost: float = float("inf")
    max_latency_ms: float = float("inf")
    require_health: bool = True
    allowed_environments: Tuple[str, ...] = ("staging", "production")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reasons: Tuple[str, ...] = ()


class AWEAIPolicyEngine:
    """Fail-closed policy evaluator for production operations."""

    def evaluate(self, policy: Policy, *, environment: str, cost: float,
                 latency_ms: float, healthy: bool) -> PolicyDecision:
        reasons: List[str] = []
        if environment not in policy.allowed_environments:
            reasons.append("environment")
        if cost < 0 or cost > policy.max_cost:
            reasons.append("cost")
        if latency_ms < 0 or latency_ms > policy.max_latency_ms:
            reasons.append("latency")
        if policy.require_health and not healthy:
            reasons.append("health")
        return PolicyDecision(not reasons, tuple(reasons))


@dataclass(frozen=True)
class ExecutionCandidate:
    name: str
    cost: float
    latency_ms: float
    quality: float
    healthy: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    selected: str
    reason: str
    fingerprint: str


class AWEAIExecutionPlanner:
    """Deterministically choose the best eligible execution candidate."""

    def plan(self, candidates: Sequence[ExecutionCandidate], policy: Policy) -> Optional[ExecutionPlan]:
        eligible = [c for c in candidates if c.healthy and c.cost <= policy.max_cost and c.latency_ms <= policy.max_latency_ms]
        if not eligible:
            return None
        # Prefer quality, then latency, then cost, then stable name.
        chosen = max(eligible, key=lambda c: (c.quality, -c.latency_ms, -c.cost, c.name))
        payload = {"candidate": chosen.__dict__, "policy": policy.__dict__}
        return ExecutionPlan(chosen.name, "quality>latency>cost>name", _digest(payload))


@dataclass(frozen=True)
class Observation:
    timestamp: float
    service: str
    status: str
    latency_ms: float
    error_rate: float
    metadata_digest: str


class AWEAIObservability:
    """Small in-process observation store suitable for deterministic tests."""

    def __init__(self) -> None:
        self._events: List[Observation] = []

    def record(self, service: str, *, status: str, latency_ms: float,
               error_rate: float, metadata: Optional[Mapping[str, Any]] = None) -> Observation:
        if latency_ms < 0 or not 0 <= error_rate <= 1:
            raise ValueError("invalid observation metrics")
        event = Observation(time.time(), service, status, latency_ms, error_rate, _digest(metadata or {}))
        self._events.append(event)
        return event

    def snapshot(self, service: Optional[str] = None) -> Tuple[Observation, ...]:
        if service is None:
            return tuple(self._events)
        return tuple(e for e in self._events if e.service == service)


@dataclass(frozen=True)
class AuditRecord:
    action: str
    actor: str
    resource: str
    result: str
    digest: str


class AWEAIAuditLog:
    """Append-only audit records with deterministic payload digests."""

    def __init__(self) -> None:
        self._records: List[AuditRecord] = []

    def append(self, *, action: str, actor: str, resource: str, result: str) -> AuditRecord:
        payload = {"action": action, "actor": actor, "resource": resource, "result": result, "index": len(self._records)}
        record = AuditRecord(action, actor, resource, result, _digest(payload))
        self._records.append(record)
        return record

    def records(self) -> Tuple[AuditRecord, ...]:
        return tuple(self._records)


__all__ = [
    "Policy", "PolicyDecision", "AWEAIPolicyEngine",
    "ExecutionCandidate", "ExecutionPlan", "AWEAIExecutionPlanner",
    "Observation", "AWEAIObservability", "AuditRecord", "AWEAIAuditLog",
]
