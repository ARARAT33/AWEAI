"""
NEXUS CORE - Advanced Autonomous AI System
==========================================
A comprehensive AI framework capable of autonomous computer operations,
intelligent decision-making, and self-improving algorithms.

Author: Nexus AI Development Team
Version: 1.0.0
"""

from .core.engine import NexusEngine
from .brain.cognitive_processor import CognitiveProcessor
from .automation.task_executor import TaskExecutor
from .learning.adaptive_learner import AdaptiveLearner
from .perception.multi_modal_analyzer import MultiModalAnalyzer
from .memory.knowledge_base import KnowledgeBase

__version__ = "1.0.0"
__author__ = "Nexus AI"
__all__ = [
    "NexusEngine",
    "CognitiveProcessor",
    "TaskExecutor",
    "AdaptiveLearner",
    "MultiModalAnalyzer",
    "KnowledgeBase"
]