"""Tests for deterministic AWEAI capability coverage."""
from aweai.capability_matrix import CapabilityMatrix


def test_matrix_is_machine_readable_and_reaches_product_target():
    matrix = CapabilityMatrix()
    report = matrix.report()
    assert 0.0 <= report["overall"] <= 1.0
    assert report["overall_percent"] >= 50.0
    assert len(report["capabilities"]) >= 15
    assert "domains" in report


def test_matrix_gate_is_deterministic():
    a = CapabilityMatrix().report()
    b = CapabilityMatrix().report()
    assert a == b
    assert CapabilityMatrix().gate(0.50) is True
