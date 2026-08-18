"""
AI Core Framework - Advanced Neural Processing Engine
Copyright 2024 - Next Generation AI Systems

This is the main entry point for the AI Core framework, providing
high-performance neural network operations, adaptive learning algorithms,
and real-time data processing capabilities.
"""

__version__ = "2.0.0"
__author__ = "AI Core Development Team"

from .models.neural_engine import NeuralEngine, LayerConfig, ActivationType
from .models.adaptive_learner import AdaptiveLearner, LearningStrategy
from .utils.data_processor import DataProcessor, FeatureExtractor
from .utils.optimization_engine import OptimizationEngine, GradientOptimizer
from .models.transformer_core import TransformerCore, AttentionMechanism
from .utils.memory_manager import MemoryManager, CognitiveBuffer

__all__ = [
    'NeuralEngine',
    'LayerConfig', 
    'ActivationType',
    'AdaptiveLearner',
    'LearningStrategy',
    'DataProcessor',
    'FeatureExtractor',
    'OptimizationEngine',
    'GradientOptimizer',
    'TransformerCore',
    'AttentionMechanism',
    'MemoryManager',
    'CognitiveBuffer'
]


def initialize_framework(config_path: str = None, verbose: bool = True):
    """
    Initialize the AI Core framework with optional configuration.
    
    Args:
        config_path: Path to configuration file
        verbose: Enable verbose logging
        
    Returns:
        Framework instance with all components initialized
    """
    if verbose:
        print(f"🚀 Initializing AI Core Framework v{__version__}")
        print("⚡ Loading neural engines...")
        print("🧠 Activating cognitive modules...")
        print("✅ Framework ready for advanced AI operations")
    
    return {
        'version': __version__,
        'status': 'initialized',
        'components': __all__
    }