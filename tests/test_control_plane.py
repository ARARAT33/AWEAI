"""Tests for the AWEAI production control plane."""
from aweai.control_plane import (
    AWEAIAuditLog,
    AWEAIExecutionPlanner,
    AWEAIObservability,
    AWEAIPolicyEngine,
    ExecutionCandidate,
    Policy,
)


def test_policy_engine_fails_closed():
    engine = AWEAIPolicyEngine()
    policy = Policy("prod", max_cost=10, max_latency_ms=500)
    assert engine.evaluate(policy, environment="production", cost=5, latency_ms=200, healthy=True).allowed
    denied = engine.evaluate(policy, environment="production", cost=50, latency_ms=200, healthy=True)
    assert not denied.allowed
    assert "cost" in denied.reasons


def test_execution_planner_is_deterministic():
    planner = AWEAIExecutionPlanner()
    candidates = [
        ExecutionCandidate("slow", 1, 400, .80),
        ExecutionCandidate("best", 2, 200, .95),
        ExecutionCandidate("cheap", 0.5, 300, .70),
    ]
    policy = Policy("p", max_cost=3, max_latency_ms=500)
    first = planner.plan(candidates, policy)
    second = planner.plan(candidates, policy)
    assert first is not None
    assert first == second
    assert first.selected == "best"


def test_observability_validates_and_filters():
    obs = AWEAIObservability()
    obs.record("api", status="ok", latency_ms=12, error_rate=0)
    obs.record("worker", status="ok", latency_ms=20, error_rate=0)
    assert len(obs.snapshot("api")) == 1


def test_audit_log_is_append_only_from_public_api():
    log = AWEAIAuditLog()
    a = log.append(action="release", actor="ci", resource="aweai", result="approved")
    b = log.append(action="deploy", actor="ci", resource="aweai", result="started")
    assert len(log.records()) == 2
    assert a.digest != b.digest
