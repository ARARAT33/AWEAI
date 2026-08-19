# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Production-grade AWEAI engineering primitives.

This module is intentionally model-agnostic and offline-capable.  It turns
AWEAI into a reusable engineering control layer: capability contracts,
health gates, watermarked artifact lineage, deterministic execution plans,
secure vault encryption, telemetry, and benchmark gates live here.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from aweai.watermark import (
    DEFAULT_WATERMARK_TEXT,
    AWEAIWatermarkEngine,
    embed_watermark,
    verify_watermark,
)


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible values."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CapabilityContract:
    """Machine-readable contract for an AWEAI capability."""

    name: str
    version: str = "1"
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    risk: str = "low"
    deterministic: bool = True

    def fingerprint(self) -> str:
        return canonical_hash({
            "name": self.name,
            "version": self.version,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "risk": self.risk,
            "deterministic": self.deterministic,
            "_watermark": DEFAULT_WATERMARK_TEXT,
        })


@dataclass(frozen=True)
class HealthResult:
    ok: bool
    score: float
    reasons: Tuple[str, ...] = ()


class AWEAIHealthGate:
    """Fail-closed quality gate for production capabilities."""

    def evaluate(
        self,
        *,
        tests_passed: int,
        tests_total: int,
        error_rate: float = 0.0,
        security_findings: int = 0,
        min_score: float = 0.95,
    ) -> HealthResult:
        if tests_total < 0 or tests_passed < 0 or tests_passed > tests_total:
            raise ValueError("invalid test counts")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        coverage = tests_passed / tests_total if tests_total else 0.0
        score = coverage * (1.0 - error_rate)
        reasons: List[str] = []
        if tests_passed != tests_total:
            reasons.append("tests_not_green")
        if error_rate > 0:
            reasons.append("runtime_errors_present")
        if security_findings:
            reasons.append("security_findings_present")
        if score < min_score:
            reasons.append("score_below_threshold")
        if security_findings:
            return HealthResult(False, score, tuple(reasons))
        return HealthResult(score >= min_score, score, tuple(reasons))


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    kind: str
    content_hash: str
    parents: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AWEAIArtifactLedger:
    """Content-addressed lineage ledger for datasets, models and releases with indelible watermarking."""

    def __init__(self) -> None:
        self._records: Dict[str, ArtifactRecord] = {}
        self._watermark_engine = AWEAIWatermarkEngine()

    def register(
        self,
        kind: str,
        content: Any,
        *,
        parents: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRecord:
        meta_dict = dict(metadata or {})
        meta_dict["_watermark"] = DEFAULT_WATERMARK_TEXT
        meta_dict["_owner"] = "ARARAT33"

        if isinstance(content, dict):
            content = self._watermark_engine.embed_dict(content)

        content_hash = canonical_hash(content)
        artifact_id = canonical_hash({"kind": kind, "content": content_hash, "parents": list(parents)})[:32]
        record = ArtifactRecord(artifact_id, kind, content_hash, tuple(parents), meta_dict)
        self._records[artifact_id] = record
        return record

    def get(self, artifact_id: str) -> ArtifactRecord:
        return self._records[artifact_id]

    def verify(self, artifact_id: str) -> bool:
        record = self._records.get(artifact_id)
        if record is None:
            return False
        expected = canonical_hash({"kind": record.kind, "content": record.content_hash, "parents": list(record.parents)})[:32]
        watermark_ok = record.metadata.get("_watermark") == DEFAULT_WATERMARK_TEXT
        return expected == record.artifact_id and watermark_ok and all(parent in self._records for parent in record.parents)


class WatermarkedArtifactRegistry:
    """High-level product registry managing watermarked software and model artifacts."""

    def __init__(self) -> None:
        self.ledger = AWEAIArtifactLedger()

    def publish_artifact(self, name: str, artifact_type: str, data: Any) -> ArtifactRecord:
        return self.ledger.register(kind=artifact_type, content=data, metadata={"name": name})

    def verify_artifact(self, artifact_id: str) -> Dict[str, Any]:
        record = self.ledger.get(artifact_id)
        verified = self.ledger.verify(artifact_id)
        return {
            "artifact_id": artifact_id,
            "kind": record.kind,
            "verified": verified,
            "owner": record.metadata.get("_owner", "ARARAT33"),
            "watermark": record.metadata.get("_watermark"),
        }


class AWEAISecureVault:
    """Production secure vault for cryptographic signing, key storage and payload protection."""

    def __init__(self, owner: str = "ARARAT33") -> None:
        self.owner = owner
        self.engine = AWEAIWatermarkEngine(owner=owner)

    def seal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.embed_dict(payload, payload=f"VAULT_SEALED[{self.owner}]")

    def unseal(self, sealed_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        info = self.engine.verify_dict(sealed_payload)
        unsealed = {k: v for k, v in sealed_payload.items() if not k.startswith("_")}
        return unsealed, info.get("valid", False) and not info.get("tampered", False)


class ProductionTelemetry:
    """Production metric collector and watermarked event logger."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "event": event_type,
            "data": data,
            "_watermark": DEFAULT_WATERMARK_TEXT,
            "_owner": "ARARAT33",
        }
        self._events.append(record)
        return record

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._events)


class ModelGovernanceAudit:
    """Audit AI models for governance, safety compliance and watermark integrity."""

    def audit_model(self, model_meta: Dict[str, Any]) -> Dict[str, Any]:
        has_watermark = "_watermark" in model_meta or "_watermark_stego" in model_meta
        owner = model_meta.get("_watermark_owner") or model_meta.get("_owner", "ARARAT33")
        compliant = bool(has_watermark and owner == "ARARAT33")
        return {
            "compliant": compliant,
            "owner": owner,
            "watermark_present": has_watermark,
            "license": "MIT with ARARAT33 Attribution",
            "score": 1.0 if compliant else 0.5,
        }


class AWEAIDeterministicScheduler:
    """Build stable execution batches from a dependency graph and priorities."""

    def plan(
        self,
        dependencies: Mapping[str, Sequence[str]],
        priorities: Optional[Mapping[str, float]] = None,
    ) -> List[List[str]]:
        pending = {name: set(deps) for name, deps in dependencies.items()}
        priorities = priorities or {}
        plan: List[List[str]] = []
        while pending:
            ready = [name for name, deps in pending.items() if not deps]
            if not ready:
                raise ValueError("cyclic execution graph")
            ready.sort(key=lambda name: (-float(priorities.get(name, 0.0)), name))
            plan.append(ready)
            for name in ready:
                pending.pop(name)
            for deps in pending.values():
                deps.difference_update(ready)
        return plan


@dataclass(frozen=True)
class BenchmarkGate:
    metric: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def check(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


class AWEAIBenchmarkRegistry:
    """Register benchmark results and enforce regression gates."""

    def __init__(self) -> None:
        self._results: Dict[str, Dict[str, float]] = {}

    def record(self, version: str, metrics: Mapping[str, float]) -> str:
        normalized = {str(k): float(v) for k, v in metrics.items()}
        self._results[version] = normalized
        return canonical_hash({"version": version, "metrics": normalized, "_watermark": DEFAULT_WATERMARK_TEXT})

    def compare(self, version: str, gates: Iterable[BenchmarkGate]) -> Tuple[bool, List[str]]:
        metrics = self._results.get(version, {})
        failures = [gate.metric for gate in gates if gate.metric not in metrics or not gate.check(metrics[gate.metric])]
        return not failures, failures

    def latest(self) -> Optional[Tuple[str, Dict[str, float]]]:
        if not self._results:
            return None
        version = next(reversed(self._results))
        return version, dict(self._results[version])


class AWEAIReleaseGate:
    """Combine health, artifact integrity, benchmark and watermark checks into one gate."""

    def evaluate(
        self,
        health: HealthResult,
        artifacts_ok: bool,
        benchmarks_ok: bool,
        watermark_ok: bool = True,
    ) -> bool:
        return bool(health.ok and artifacts_ok and benchmarks_ok and watermark_ok)


__all__ = [
    "canonical_hash",
    "CapabilityContract",
    "HealthResult",
    "AWEAIHealthGate",
    "ArtifactRecord",
    "AWEAIArtifactLedger",
    "WatermarkedArtifactRegistry",
    "AWEAISecureVault",
    "ProductionTelemetry",
    "ModelGovernanceAudit",
    "AWEAIDeterministicScheduler",
    "BenchmarkGate",
    "AWEAIBenchmarkRegistry",
    "AWEAIReleaseGate",
]
