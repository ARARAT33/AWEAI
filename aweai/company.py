# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Universal AI-company tooling registry and AWEAI-only control plane.

This module deliberately describes *capabilities*, not autonomous agents.  AWEAI
is the engineering/control-plane layer: models, vendors and infrastructure are
implementation details behind normalized capabilities.

The registry is intentionally data-driven so new capabilities can be added
without creating a new Python command for every individual task.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from aweai.cmd.common import APP_DIR

REGISTRY_VERSION = "1.1"

CAPABILITY_CATALOG: Dict[str, List[str]] = {
    "research": [
        "literature_search", "paper_ingest", "citation_graph", "benchmark_discovery",
        "experiment_design", "hypothesis_tracking", "knowledge_extraction", "technical_survey",
        "prior_art", "reproducibility", "research_notes", "dataset_discovery",
    ],
    "data": [
        "crawl", "scrape", "import", "export", "normalize", "deduplicate", "clean",
        "filter", "label", "annotate", "augment", "synthetic_data", "split", "shuffle",
        "tokenize", "detokenize", "embed", "rerank", "index", "version", "lineage",
        "quality", "bias_audit", "pii_detection", "redaction", "provenance", "schema",
        "parquet", "jsonl", "csv", "image_dataset", "audio_dataset", "video_dataset",
    ],
    "model": [
        "architecture", "initialize", "train", "pretrain", "continue_train", "finetune",
        "instruction_tune", "preference_tune", "distill", "prune", "quantize", "merge",
        "convert", "export", "import", "compile", "profile", "benchmark", "evaluate",
        "regression_test", "safety_eval", "robustness_eval", "interpretability", "calibrate",
        "checkpoint", "resume", "reproduce", "model_card", "registry", "versioning",
    ],
    "multimodal": [
        "text", "vision", "image_generation", "image_editing", "ocr", "audio", "speech",
        "transcription", "translation", "video", "video_analysis", "video_generation",
        "3d", "document_ai", "realtime", "streaming", "cross_modal_embedding",
    ],
    "engineering": [
        "project_init", "scaffold", "code_generation", "code_edit", "refactor", "lint",
        "format", "typecheck", "static_analysis", "dependency_audit", "build", "package",
        "release", "migration", "api_design", "sdk_generation", "schema_generation",
        "documentation", "changelog", "examples", "test_generation", "unit_test",
        "integration_test", "e2e_test", "fuzz_test", "load_test", "debug", "profiling",
    ],
    "devops": [
        "environment", "secrets", "artifact", "cache", "ci", "cd", "deployment", "rollback",
        "canary", "blue_green", "autoscaling", "healthcheck", "logs", "metrics", "traces",
        "alerting", "incident", "backup", "restore", "disaster_recovery", "capacity",
        "cost_control", "resource_scheduling", "gpu_scheduling", "cluster", "ssh", "remote_exec",
    ],
    "infrastructure": [
        "cpu", "gpu", "npu", "memory", "storage", "network", "container", "vm", "bare_metal",
        "distributed_training", "parallelism", "checkpoint_storage", "object_storage", "database",
        "vector_database", "queue", "cache_store", "scheduler", "edge", "local_inference",
    ],
    "product": [
        "requirements", "specification", "roadmap", "prioritization", "architecture", "prototype",
        "feature_flags", "experiments", "ab_testing", "release_planning", "telemetry", "feedback",
        "usability", "accessibility", "localization", "internationalization", "deprecation",
    ],
    "security": [
        "threat_model", "secret_scan", "dependency_scan", "sast", "dast", "container_scan",
        "permission_audit", "identity", "authentication", "authorization", "least_privilege",
        "sandbox", "isolation", "policy", "audit_log", "tamper_detection", "supply_chain",
        "prompt_injection_defense", "data_exfiltration_defense", "model_abuse_detection",
    ],
    "governance": [
        "policy", "approval", "audit", "provenance", "data_retention", "access_review",
        "model_risk", "vendor_risk", "compliance", "license_review", "attribution", "privacy",
        "safety_case", "change_control", "release_gate", "evidence_bundle",
    ],
    "operations": [
        "users", "teams", "roles", "permissions", "workspaces", "projects", "jobs", "queues",
        "workflows", "schedules", "cron", "retries", "timeouts", "cancellation", "idempotency",
        "priority", "resource_budget", "latency_budget", "usage_tracking", "billing_metering",
        "quota", "rate_limit", "service_catalog", "runbook", "status", "health",
    ],
    "inference": [
        "routing", "fallback", "load_balance", "batch", "stream", "cache", "speculative",
        "dynamic_model_selection", "context_management", "structured_output", "tool_contract",
        "latency_optimization", "cost_optimization", "quality_optimization", "offline_inference",
        "edge_inference", "server_inference", "model_warmup", "concurrency",
    ],
    "knowledge": [
        "knowledge_base", "rag", "retrieval", "chunking", "indexing", "metadata", "citation",
        "memory", "semantic_search", "hybrid_search", "knowledge_graph", "entity_resolution",
        "fact_check", "freshness", "source_ranking", "document_ingestion", "workspace_memory",
    ],
    "business": [
        "market_research", "competitive_analysis", "pricing", "unit_economics", "forecasting",
        "capacity_planning", "customer_analysis", "funnel", "retention", "growth", "seo",
        "content", "campaign", "support", "feedback_analysis", "sales_operations", "reporting",
    ],
    "platform": [
        "cli", "api", "plugin", "adapter", "connector", "registry", "catalog", "discovery",
        "capability_negotiation", "healthcheck", "versioning", "compatibility", "telemetry",
        "observability", "feature_registry", "policy_engine", "execution_gateway", "trace_store",
    ],
}


def all_capabilities() -> List[str]:
    """Return normalized ``category.capability`` names in stable order."""
    return [f"{category}.{name}" for category, names in CAPABILITY_CATALOG.items() for name in names]


@dataclass(frozen=True)
class Capability:
    name: str
    category: str
    implementation: str = "aweai"
    executable: bool = True
    requires_external_adapter: bool = False


class AWEAIOnlyPolicy:
    """Local policy object preventing direct vendor execution in this layer."""

    ALLOWED_PATHS = frozenset({"aweai", "aweai.gateway", "aweai.adapter"})

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @property
    def path(self) -> Path:
        return APP_DIR / "aweai_only.json"

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"enabled": self.enabled, "version": REGISTRY_VERSION}, indent=2),
            encoding="utf-8",
        )
        return self.path

    def assert_execution_path(self, path: str) -> None:
        if self.enabled and path not in self.ALLOWED_PATHS:
            raise PermissionError("AWEAI-only policy blocks direct tool execution")


class CompanyToolRegistry:
    """Discover, search and validate the complete AWEAI company capability surface."""

    def __init__(self) -> None:
        self.policy = AWEAIOnlyPolicy(True)
        self._index: Optional[Dict[str, Capability]] = None

    def capabilities(self, category: Optional[str] = None) -> List[Capability]:
        rows: List[Capability] = []
        for cat, names in CAPABILITY_CATALOG.items():
            if category and cat != category:
                continue
            rows.extend(Capability(name=f"{cat}.{n}", category=cat) for n in names)
        return rows

    def _build_index(self) -> Dict[str, Capability]:
        return {row.name: row for row in self.capabilities()}

    @property
    def index(self) -> Dict[str, Capability]:
        """Return a cached capability lookup index."""
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def get(self, capability: str) -> Optional[Capability]:
        """Resolve an exact ``category.capability`` identifier."""
        return self.index.get(capability.strip().lower())

    def search(self, query: str, category: Optional[str] = None, limit: int = 25) -> List[Capability]:
        """Search capability IDs by case-insensitive substring."""
        q = query.strip().lower()
        if not q:
            return []
        rows = [row for row in self.capabilities(category) if q in row.name.lower()]
        return rows[: max(0, limit)]

    def category_stats(self) -> Dict[str, int]:
        """Return capability counts per company-engineering domain."""
        return {category: len(names) for category, names in CAPABILITY_CATALOG.items()}

    def execution_plan(self, capability: str, *, adapter: Optional[str] = None) -> dict:
        """Build a safe, non-executing routing plan for a capability."""
        row = self.get(capability)
        if row is None:
            return {"ok": False, "error": f"unknown capability: {capability}"}
        self.policy.assert_execution_path("aweai.adapter" if adapter else "aweai")
        return {
            "ok": True,
            "capability": row.name,
            "control_plane": "AWEAI",
            "execution_path": "aweai.adapter" if adapter else "aweai",
            "adapter": adapter,
            "requires_external_adapter": bool(adapter),
            "executable_here": row.executable and not bool(adapter),
            "dry_run": True,
        }

    def manifest(self) -> dict:
        rows = self.capabilities()
        return {
            "name": "AWEAI Universal AI Company Tooling",
            "version": REGISTRY_VERSION,
            "mode": "aweai_only",
            "categories": len(CAPABILITY_CATALOG),
            "capabilities": len(rows),
            "category_stats": self.category_stats(),
            "capability_ids": [r.name for r in rows],
            "execution": {
                "control_plane": "AWEAI",
                "direct_vendor_tools": False,
                "vendor_apis": "adapter implementation detail",
                "requires_credentials_when_external": True,
            },
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate(self) -> dict:
        rows = self.capabilities()
        names = [r.name for r in rows]
        duplicate_set = sorted({n for n in names if names.count(n) > 1})
        malformed = sorted(n for n in names if n.count(".") != 1 or n.split(".")[0] not in CAPABILITY_CATALOG)
        empty_categories = sorted(c for c, values in CAPABILITY_CATALOG.items() if not values)
        return {
            "ok": not duplicate_set and not malformed and not empty_categories and bool(rows),
            "capabilities": len(rows),
            "categories": len(CAPABILITY_CATALOG),
            "duplicates": duplicate_set,
            "malformed": malformed,
            "empty_categories": empty_categories,
            "fingerprint": self.fingerprint(),
            "policy": {"mode": "aweai_only", "direct_vendor_tools": False},
        }


def manifest_json() -> str:
    return json.dumps(CompanyToolRegistry().manifest(), indent=2, ensure_ascii=False)
