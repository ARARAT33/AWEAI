from aweai.product import (
    AWEAIArtifactLedger,
    AWEAIBenchmarkRegistry,
    AWEAIHealthGate,
    AWEAIReleaseGate,
    AWEAIDeterministicScheduler,
    BenchmarkGate,
    CapabilityContract,
)


def test_capability_contract_is_stable():
    contract = CapabilityContract("training", version="2", inputs=("dataset",), outputs=("model",))
    assert len(contract.fingerprint()) == 64
    assert contract.fingerprint() == contract.fingerprint()


def test_health_gate_fails_closed():
    gate = AWEAIHealthGate()
    assert gate.evaluate(tests_passed=10, tests_total=10).ok
    assert not gate.evaluate(tests_passed=9, tests_total=10).ok
    assert not gate.evaluate(tests_passed=10, tests_total=10, security_findings=1).ok


def test_artifact_lineage_and_integrity():
    ledger = AWEAIArtifactLedger()
    dataset = ledger.register("dataset", {"rows": 10})
    model = ledger.register("model", {"weights": "abc"}, parents=(dataset.artifact_id,))
    assert ledger.verify(dataset.artifact_id)
    assert ledger.verify(model.artifact_id)


def test_scheduler_is_deterministic_and_priority_aware():
    scheduler = AWEAIDeterministicScheduler()
    graph = {"build": [], "test": ["build"], "package": ["test"], "lint": ["build"]}
    assert scheduler.plan(graph, {"lint": 10}) == [["build"], ["lint", "test"], ["package"]]


def test_benchmark_regression_gate():
    registry = AWEAIBenchmarkRegistry()
    registry.record("1.0", {"latency_ms": 20, "accuracy": 0.98})
    ok, failures = registry.compare("1.0", [BenchmarkGate("accuracy", minimum=0.97), BenchmarkGate("latency_ms", maximum=25)])
    assert ok and failures == []
    ok, failures = registry.compare("1.0", [BenchmarkGate("accuracy", minimum=0.99)])
    assert not ok and failures == ["accuracy"]


def test_release_gate_requires_everything():
    health = AWEAIHealthGate().evaluate(tests_passed=10, tests_total=10)
    gate = AWEAIReleaseGate()
    assert gate.evaluate(health, artifacts_ok=True, benchmarks_ok=True)
    assert not gate.evaluate(health, artifacts_ok=False, benchmarks_ok=True)
