# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Unit tests for Genesis Core algorithms."""

import pytest
from genesis_core import (
    ChaosEncoder,
    CyberKey,
    EmotionalAnalyzer,
    FractalMemoryNode,
    GeneticPoet,
    HolographicCompressor,
    HyperDimensionalVectorEngine,
    NeuroPlasticGraph,
    QuantumFourierTransformSim,
    QuantumState,
    StochasticSolver,
    TemporalBuffer,
    run_genesis_demonstration,
)


def test_quantum_state_and_qft():
    q1 = QuantumState(100.0, phase=0.0)
    q2 = QuantumState(200.0, phase=0.5)
    q1.entangle(q2)
    assert q1.entropy == q2.entropy
    q1.evolve()
    res = q1.measure()
    assert isinstance(res, (int, float))

    qft = QuantumFourierTransformSim(num_qubits=2)
    state = [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]
    freq = qft.transform(state)
    assert len(freq) == 4
    recovered = qft.inverse_transform(freq)
    assert abs(recovered[0].real - 1.0) < 1e-5


def test_neuro_plastic_graph():
    g = NeuroPlasticGraph()
    for i in range(5):
        g.add_node(f"N{i}")
    for i in range(4):
        g.connect(f"N{i}", f"N{i+1}", initial_weight=1.0)

    path = g.traverse_adaptive("N0", steps=5)
    assert len(path) > 0
    opt_path = g.find_optimal_synaptic_path("N0", "N4")
    assert opt_path == ["N0", "N1", "N2", "N3", "N4"]

    pruned = g.prune_synapses(threshold=0.01)
    assert isinstance(pruned, int)


def test_chaos_encoder():
    lorenz = ChaosEncoder(system_type="lorenz")
    traj_l = lorenz.encode("Test")
    assert len(traj_l) == 4

    rossler = ChaosEncoder(system_type="rossler")
    traj_r = rossler.encode("Test")
    assert len(traj_r) == 4

    decoded = lorenz.decode_approx(traj_l)
    assert isinstance(decoded, str)


def test_hypervector_engine():
    vsa = HyperDimensionalVectorEngine(dim=1000, seed=123)
    v1 = vsa.create_vector("A")
    v2 = vsa.create_vector("B")
    bound = vsa.bind(v1, v2)
    assert len(bound) == 1000

    bundled = vsa.bundle([v1, v2])
    assert len(bundled) == 1000

    match, sim = vsa.query_memory(v1)
    assert match == "A"
    assert sim > 0.99


def test_emotional_analyzer_and_temporal_buffer():
    analyzer = EmotionalAnalyzer()
    score_pos = analyzer.analyze_sentiment("great success fast")
    assert score_pos > 0.5

    buf = TemporalBuffer(capacity=2)
    buf.add("Item1", importance=0.9, timestamp=100.0)
    buf.add("Item2", importance=0.1, timestamp=100.0)
    buf.add("Item3", importance=0.8, timestamp=100.0)
    context = buf.get_context()
    assert len(context) == 2


def test_poet_fractal_stochastic_holo_key():
    poet = GeneticPoet(["a", "b", "c"])
    line = poet.generate_line(3)
    assert len(line) == 3

    node = FractalMemoryNode("root", max_depth=2)
    node.store("hello")
    res = node.retrieve_pattern(1)
    assert isinstance(res, list)

    solver = StochasticSolver(initial_temp=10.0, cooling_rate=0.5)
    best_st, best_cost = solver.solve([1, 2, 3], lambda st: sum(st))
    assert isinstance(best_cost, (int, float))

    holo = HolographicCompressor(block_size=4)
    blocks = holo.project("Hello World")
    assert len(blocks) > 0
    hint = holo.reconstruct_hint(blocks[0])
    assert "Global Signature Hint" in hint

    key = CyberKey("Secret")
    ok = key.attempt_access("Secret")
    assert ok is True
    status = key.get_status()
    assert status["generation"] == 1


def test_genesis_demonstration_runs():
    run_genesis_demonstration()
