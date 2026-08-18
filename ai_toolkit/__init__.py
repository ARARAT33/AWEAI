#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Toolkit - Հզոր գործիք AI մշակողների համար
Full-featured AI development toolkit with comprehensive modules

Features:
- Data Collection & Preprocessing
- Model Training & Evaluation
- Device Control & IoT
- Module Management & Plugins
- Analytics & Visualization
- Task Scheduling & Progress Tracking
"""

__version__ = "1.0.0"
__author__ = "AI Toolkit Team"

from .modules.data_collection import DataCollector, DataPreprocessor
from .modules.model_training import ModelTrainer, HyperparameterTuner, ModelEvaluator
from .modules.device_control import DeviceController, GPIOPinController
from .modules.modules import ModuleManager, PluginSystem
from .utils.helpers import Logger, ConfigManager, TaskScheduler, ProgressTracker
from .utils.analytics import DataAnalyzer, VisualizationTools, ReportGenerator

__all__ = [
    # Data Collection
    'DataCollector',
    'DataPreprocessor',
    
    # Model Training
    'ModelTrainer',
    'HyperparameterTuner',
    'ModelEvaluator',
    
    # Device Control
    'DeviceController',
    'GPIOPinController',
    
    # Module Management
    'ModuleManager',
    'PluginSystem',
    
    # Utilities
    'Logger',
    'ConfigManager',
    'TaskScheduler',
    'ProgressTracker',
    
    # Analytics
    'DataAnalyzer',
    'VisualizationTools',
    'ReportGenerator'
]
