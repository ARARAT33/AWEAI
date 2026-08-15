from aweai.orchestration import (
    AWEAIOrchestrator,
    CanaryPolicy,
    OrchestrationError,
    ResourcePool,
    RolloutState,
    Workload,
)


def _orch():
    return AWEAIOrchestrator([
        ResourcePool("cpu", cpu=8, memory_gb=32),
        ResourcePool("gpu", cpu=16, memory_gb=64, accelerator=1),
    ])


def test_deterministic_dependency_plan():
    jobs = [
        Workload("train", cpu=8, memory_gb=16, accelerator=1),
        Workload("evaluate", depends_on=("train",)),
    ]
    a = _orch().plan(jobs)
    b = _orch().plan(jobs)
    assert a == b
    assert [x.workload for x in a] == ["train", "evaluate"]


def test_cycle_is_rejected():
    try:
        _orch().plan([Workload("a", depends_on=("b",)), Workload("b", depends_on=("a",))])
    except OrchestrationError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle must be rejected")


def test_unschedulable_workload_is_rejected():
    try:
        _orch().plan([Workload("huge", cpu=100, memory_gb=1000)])
    except OrchestrationError as exc:
        assert "resource pool" in str(exc)
    else:
        raise AssertionError("unschedulable workload must be rejected")


def test_plan_fingerprint_is_stable():
    jobs = [Workload("a"), Workload("b", depends_on=("a",))]
    assert _orch().fingerprint(jobs) == _orch().fingerprint(jobs)


def test_canary_fails_closed_on_bad_health():
    policy = CanaryPolicy()
    state = RolloutState(desired=100, healthy=95, error_rate=0.01, traffic_percent=10)
    assert policy.decide(state, 25) == "rollback"


def test_canary_promotes_healthy_release():
    policy = CanaryPolicy()
    state = RolloutState(desired=100, healthy=100, error_rate=0.001, traffic_percent=10)
    assert policy.decide(state, 25) == "promote"
