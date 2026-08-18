"""Test suite for AI Fabric Core algorithms"""

import pytest
import numpy as np
from ai_fabric_core.src.algorithms.adaptive_gradient import AdaptiveGradientSync, SyncStrategy, CompressionType
from ai_fabric_core.src.algorithms.quantum_attention import QuantumAttention
from ai_fabric_core.src.algorithms.neural_evolution import NeuralArchitectureEvolution
from ai_fabric_core.src.algorithms.federated_learning import FederatedLearningDP
from ai_fabric_core.src.algorithms.multimodal_fusion import MultiModalFusionTransformer
from ai_fabric_core.src.algorithms.self_healing import SelfHealingModelSystem, HealthMetrics, HealthStatus
from ai_fabric_core.src.algorithms.resource_allocator import DynamicResourceAllocator, Workload
from ai_fabric_core.src.algorithms.model_converter import CrossPlatformModelConverter, ModelFormat


class TestAdaptiveGradientSync:
    """Tests for AGS algorithm"""
    
    def test_initialization(self):
        ags = AdaptiveGradientSync(num_nodes=4, num_workers_per_node=8)
        assert ags.num_nodes == 4
        assert ags.num_workers == 32
    
    def test_gradient_analysis(self):
        ags = AdaptiveGradientSync()
        gradients = {
            'layer1': np.random.randn(10, 10),
            'layer2': np.random.randn(5, 5),
        }
        stats = ags.analyze_gradients(gradients)
        assert stats.norm > 0
        assert 0 <= stats.sparsity <= 1
    
    def test_strategy_selection(self):
        ags = AdaptiveGradientSync()
        from ai_fabric_core.src.algorithms.adaptive_gradient import NetworkCondition, GradientStats
        
        network = NetworkCondition(bandwidth_mbps=10000, latency_ms=1.0)
        gradients = GradientStats(norm=1.0, sparsity=0.3)
        
        strategy = ags.select_optimal_strategy(network, gradients)
        assert isinstance(strategy, SyncStrategy)


class TestQuantumAttention:
    """Tests for QIA algorithm"""
    
    def test_initialization(self):
        qia = QuantumAttention(num_qubits=12, num_heads=8)
        assert qia.num_qubits == 12
        assert qia.num_heads == 8
    
    def test_forward_pass(self):
        qia = QuantumAttention(num_heads=2, hidden_dim=64)
        batch_size, seq_len, dim = 2, 10, 64
        
        query = np.random.randn(batch_size, seq_len, dim)
        key = np.random.randn(batch_size, seq_len, dim)
        value = np.random.randn(batch_size, seq_len, dim)
        
        output = qia.forward(query, key, value)
        assert output.shape == (batch_size, seq_len, dim)


class TestNeuralArchitectureEvolution:
    """Tests for NAE algorithm"""
    
    def test_initialization(self):
        nae = NeuralArchitectureEvolution(population_size=20)
        assert nae.population_size == 20
    
    def test_evolution(self):
        nae = NeuralArchitectureEvolution(population_size=10, num_generations=5)
        
        def fitness_fn(arch):
            return np.random.random()
        
        best = nae.evolve(fitness_fn)
        assert best is not None
        assert len(nae.history) == 5


class TestFederatedLearningDP:
    """Tests for FL-DP algorithm"""
    
    def test_initialization(self):
        fl = FederatedLearningDP(num_clients=50, privacy_epsilon=1.0)
        assert fl.num_clients == 50
        assert fl.privacy_epsilon == 1.0
    
    def test_privacy_tracking(self):
        fl = FederatedLearningDP(num_clients=10, clients_per_round=5)
        model_template = {'layer1': np.zeros((10, 10))}
        
        fl.initialize_clients(model_template)
        fl.train_round(round_num=1)
        
        status = fl.get_privacy_status()
        assert status['spent'] > 0


class TestMultiModalFusion:
    """Tests for MMFT algorithm"""
    
    def test_initialization(self):
        mmft = MultiModalFusionTransformer(modalities=['text', 'image'])
        assert 'text' in mmft.modalities
        assert 'image' in mmft.modalities
    
    def test_fusion(self):
        mmft = MultiModalFusionTransformer(hidden_dim=64)
        
        inputs = {
            'text': np.random.randn(2, 10, 64),
            'image': np.random.randn(2, 10, 64),
        }
        
        output = mmft.forward(inputs, fusion_strategy='attention')
        assert output.shape[0] == 2
        assert output.shape[-1] == 64


class TestSelfHealingModelSystem:
    """Tests for SHMS algorithm"""
    
    def test_initialization(self):
        shms = SelfHealingModelSystem()
        assert shms.max_error_rate == 0.05
    
    def test_health_check(self):
        shms = SelfHealingModelSystem()
        
        # Healthy metrics
        healthy_metrics = HealthMetrics(
            error_rate=0.01,
            latency_p99_ms=100,
            throughput_qps=500,
        )
        
        status = shms.check_health(healthy_metrics)
        assert status == HealthStatus.HEALTHY
        
        # Unhealthy metrics
        unhealthy_metrics = HealthMetrics(
            error_rate=0.1,
            latency_p99_ms=2000,
            throughput_qps=50,
        )
        
        status = shms.check_health(unhealthy_metrics)
        assert status != HealthStatus.HEALTHY


class TestDynamicResourceAllocator:
    """Tests for DRA algorithm"""
    
    def test_initialization(self):
        dra = DynamicResourceAllocator(total_gpus=8, total_memory_gb=640)
        assert dra.total_gpus == 8
    
    def test_workload_submission(self):
        dra = DynamicResourceAllocator(total_gpus=4, total_memory_gb=320)
        
        workload = Workload(
            id='test_1',
            type='training',
            priority=1,
            gpu_required=2,
            memory_gb=64,
            estimated_duration_sec=3600,
        )
        
        dra.submit_workload(workload)
        assert len(dra.allocations) > 0 or len(dra.pending_workloads) > 0


class TestModelConverter:
    """Tests for CPMC algorithm"""
    
    def test_initialization(self):
        cpmc = CrossPlatformModelConverter()
        formats = cpmc.get_supported_formats()
        assert 'pytorch' in formats
    
    def test_conversion_support(self):
        cpmc = CrossPlatformModelConverter()
        
        assert cpmc.can_convert(ModelFormat.PYTORCH, ModelFormat.ONNX)
        assert not cpmc.can_convert(ModelFormat.PYTORCH, ModelFormat.PYTORCH)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
