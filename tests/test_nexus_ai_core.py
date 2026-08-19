# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Unit tests for Nexus AI Core modules."""

import asyncio
import pytest
from nexus_ai_core import (
    DynamicGraphOptimizer,
    HyperConcurrentEngine,
    NexusAICore,
    OptimizationStrategy,
    QuantumResistantShield,
    SecurityLevel,
    SelfHealingMonitor,
    TensorTransformEngine,
)


def test_tensor_transform_engine():
    mat = [[1.0, 2.0], [3.0, 4.0]]
    res = TensorTransformEngine.svd_compress(mat, k=1)
    assert res["rank"] == 1
    assert "compression_ratio" in res

    q = TensorTransformEngine.quantize_matrix(mat)
    assert q["dtype"] == "int8"
    assert "scale" in q


def test_quantum_resistant_shield():
    shield = QuantumResistantShield(level=SecurityLevel.QUANTUM_RESISTANT)
    priv, pub = shield.generate_key_pair()
    assert len(priv) > 0
    assert len(pub) > 0

    data = {"test": 123}
    sig = shield.sign_payload(data, priv)
    assert "DILITHIUM5-SIG::" in sig
    assert shield.verify_signature(data, sig, priv) is True
    assert shield.verify_signature({"test": 999}, sig, priv) is False


def test_dynamic_graph_optimizer():
    opt = DynamicGraphOptimizer(strategy=OptimizationStrategy.GRAPH_FUSION)
    ops = ["conv2d", "batch_norm", "relu", "matmul", "bias_add"]
    fused = opt.analyze_graph(ops)
    assert "FUSED(conv2d, batch_norm)" in fused

    mem = opt.estimate_memory_footprint(ops)
    assert "savings_pct" in mem


def test_hyper_concurrent_engine():
    async def _test():
        engine = HyperConcurrentEngine(max_workers=4)

        async def async_task():
            return 42

        def sync_task():
            return 100

        results = await engine.process_batch([async_task(), sync_task])
        assert results == [42, 100]

    asyncio.run(_test())


def test_self_healing_monitor():
    async def _test():
        healer = SelfHealingMonitor()
        metrics = {"cpu_load": 0.50, "memory_usage": 0.99}
        ok = await healer.monitor(metrics)
        assert ok is False
        assert len(healer.repair_history) == 1

    asyncio.run(_test())


def test_nexus_ai_core_pipeline():
    async def _test():
        core = NexusAICore()
        await core.start()
        sample = [{"id": 1, "data": "a"}, {"id": 2, "data": "b"}]
        res = await core.process_pipeline(sample)
        assert res["status"] == "success"
        assert res["results_count"] == 2
        core.stop()

    asyncio.run(_test())
