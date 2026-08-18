"""
AI Fabric Core - Համապարփակ AI Ինֆրակառուցվածքի Համակարգ
"""

__version__ = "1.0.0"
__author__ = "AI Fabric Team"

from .core.orchestrator import SystemOrchestrator
from .core.resource_manager import ResourceManager
from .algorithms.adaptive_gradient import AdaptiveGradientSync
from .algorithms.quantum_attention import QuantumAttention
from .algorithms.neural_evolution import NeuralArchitectureEvolution
from .algorithms.federated_learning import FederatedLearningDP
from .algorithms.multimodal_fusion import MultiModalFusionTransformer
from .algorithms.self_healing import SelfHealingModelSystem
from .algorithms.resource_allocator import DynamicResourceAllocator
from .algorithms.model_converter import CrossPlatformModelConverter

__all__ = [
    "SystemOrchestrator",
    "ResourceManager",
    "AdaptiveGradientSync",
    "QuantumAttention",
    "NeuralArchitectureEvolution",
    "FederatedLearningDP",
    "MultiModalFusionTransformer",
    "SelfHealingModelSystem",
    "DynamicResourceAllocator",
    "CrossPlatformModelConverter",
]
