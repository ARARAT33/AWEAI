from aweai.production_ops import (
    AWEAICanaryController,
    AWEAICostOptimizer,
    AWEAIIncidentState,
    AWEAIReproducibility,
    CostBudget,
    ReleaseManifest,
    SLOGate,
)


def test_release_manifest_is_stable():
    manifest = ReleaseManifest("demo", "1.0", "artifact")
    assert manifest.fingerprint() == manifest.fingerprint()


def test_slo_gate_fails_closed():
    gate = SLOGate()
    assert gate.evaluate(1.0, 100.0, 0.0)[0]
    assert not gate.evaluate(0.9, 100.0, 0.0)[0]


def test_canary_controller():
    controller = AWEAICanaryController()
    decision = controller.decide(
        baseline=SLOGate(),
        canary_metrics={"availability": 1.0, "p95_latency_ms": 100.0, "error_rate": 0.0},
    )
    assert decision.promote and decision.traffic_percent == 5.0


def test_canary_blocks_bad_metrics():
    decision = AWEAICanaryController().decide(
        baseline=SLOGate(),
        canary_metrics={"availability": 0.8, "p95_latency_ms": 100.0, "error_rate": 0.0},
    )
    assert not decision.promote and decision.traffic_percent == 0.0


def test_cost_optimizer_prefers_low_cost_then_quality():
    selected = AWEAICostOptimizer().choose(
        {"a": {"cost": 2, "score": 0.9}, "b": {"cost": 1, "score": 0.8}}, budget=2
    )
    assert selected == "b"


def test_budget():
    assert CostBudget(10).check(10)
    assert not CostBudget(10).check(10.01)


def test_reproducibility_fingerprint_changes_with_inputs():
    a = AWEAIReproducibility.fingerprint(code="a", config="b", data="c")
    b = AWEAIReproducibility.fingerprint(code="a", config="x", data="c")
    assert a != b


def test_incident_state_machine():
    incident = AWEAIIncidentState()
    assert incident.transition("acknowledged") == "acknowledged"
    assert incident.transition("mitigating") == "mitigating"
    assert incident.transition("resolved") == "resolved"
