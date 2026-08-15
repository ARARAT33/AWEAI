"""Tests for deterministic AWEAI capability coverage."""
import pytest

from aweai.capability_matrix import Capability, CapabilityMatrix


def test_matrix_is_machine_readable_and_reaches_product_target():
    matrix = CapabilityMatrix()
    report = matrix.report()
    assert 0.0 <= report["overall"] <= 1.0
    assert report["overall_percent"] >= 50.0
    assert len(report["capabilities"]) >= 15
    assert "domains" in report
    assert "weakest" in report
    assert all("status" in item for item in report["capabilities"])


def test_matrix_gate_is_deterministic():
    a = CapabilityMatrix().report()
    b = CapabilityMatrix().report()
    assert a == b
    assert CapabilityMatrix().gate(0.50) is True


def test_gate_diagnostics_explain_failures():
    diagnostics = CapabilityMatrix().gate_diagnostics(0.80)
    assert diagnostics["overall_pass"] is False
    assert diagnostics["passed"] is False
    assert diagnostics["below_floor"]


def test_weakest_is_sorted_and_bounded():
    weakest = CapabilityMatrix().weakest(3)
    assert len(weakest) == 3
    assert weakest == sorted(weakest, key=lambda item: (item["score"], item["name"]))


def test_capability_validation_rejects_invalid_scores():
    with pytest.raises(ValueError):
        Capability("bad", "test", 1.1, 0.5, 0.5)

    with pytest.raises(ValueError):
        Capability("", "test", 0.5, 0.5, 0.5)
