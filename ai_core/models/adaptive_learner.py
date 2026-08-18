"""
Adaptive Learner - Intelligent Learning Rate Optimization
Implements advanced learning strategies with automatic hyperparameter tuning,
curriculum learning, and meta-learning capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import json
from collections import deque


class LearningStrategy(Enum):
    """Available learning strategies for adaptive optimization."""
    SGD = "sgd"
    MOMENTUM = "momentum"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    NADAM = "nadam"
    ADAMW = "adamw"


class AdaptiveLearner:
    """
    Advanced adaptive learning system with intelligent optimization:
    - Multiple optimization strategies
    - Learning rate scheduling
    - Gradient clipping
    - Weight decay
    - Warm restarts
    - Curriculum learning
    - Meta-learning adaptation
    """
    
    def __init__(
        self,
        strategy: LearningStrategy = LearningStrategy.ADAM,
        base_learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        warmup_steps: int = 100,
        cooldown_steps: int = 50
    ):
        self.strategy = strategy
        self.base_lr = base_learning_rate
        self.current_lr = base_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        
        # Optimizer state
        self.momentums = {}
        self.velocities = {}
        self.step_count = 0
        
        # Learning rate schedule
        self.lr_schedule = []
        self.loss_history = deque(maxlen=100)
        self.performance_trend = []
        
        # Curriculum learning state
        self.curriculum_stage = 0
        self.difficulty_level = 0.0
        
        # Meta-learning parameters
        self.meta_gradients = {}
        self.adaptation_rate = 0.1
        
    def _clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Clip gradients to prevent exploding gradients."""
        norm = np.linalg.norm(gradients)
        if norm > self.gradient_clip:
            return gradients * (self.gradient_clip / norm)
        return gradients
    
    def _get_learning_rate(self) -> float:
        """Calculate current learning rate based on schedule."""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.step_count + 1) / self.warmup_steps
            lr = self.base_lr * warmup_factor
        elif self.step_count > self.warmup_steps + self.cooldown_steps:
            # Cosine annealing after warmup
            progress = (self.step_count - self.warmup_steps) / self.cooldown_steps
            lr = self.base_lr * (1 + np.cos(np.pi * min(progress, 1))) / 2
        else:
            lr = self.base_lr
        
        # Apply learning rate decay based on loss trend
        if len(self.loss_history) > 10:
            recent_avg = np.mean(list(self.loss_history)[-10:])
            older_avg = np.mean(list(self.loss_history)[-20:-10])
            
            if recent_avg > older_avg * 1.1:  # Loss increasing
                lr *= 0.5
            elif recent_avg < older_avg * 0.9:  # Loss decreasing well
                lr *= 1.1
        
        self.current_lr = lr
        self.lr_schedule.append(lr)
        return lr
    
    def optimize(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: str = "default"
    ) -> np.ndarray:
        """
        Apply optimization strategy to update parameters.
        
        Args:
            params: Current parameter values
            gradients: Computed gradients
            param_id: Unique identifier for parameter group
            
        Returns:
            Updated parameters
        """
        # Clip gradients
        gradients = self._clip_gradients(gradients)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients += self.weight_decay * params
        
        # Get current learning rate
        lr = self._get_learning_rate()
        self.step_count += 1
        
        # Initialize optimizer state if needed
        if param_id not in self.momentums:
            self.momentums[param_id] = np.zeros_like(params)
            self.velocities[param_id] = np.zeros_like(params)
        
        m = self.momentums[param_id]
        v = self.velocities[param_id]
        
        # Apply optimization strategy
        if self.strategy == LearningStrategy.SGD:
            updates = -lr * gradients
            
        elif self.strategy == LearningStrategy.MOMENTUM:
            m = self.beta1 * m - lr * gradients
            updates = m
            
        elif self.strategy == LearningStrategy.ADAM:
            m = self.beta1 * m + (1 - self.beta1) * gradients
            v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            updates = -lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
        elif self.strategy == LearningStrategy.RMSPROP:
            v = 0.9 * v + 0.1 * (gradients ** 2)
            updates = -lr * gradients / (np.sqrt(v) + self.epsilon)
            
        elif self.strategy == LearningStrategy.ADAGRAD:
            v += gradients ** 2
            updates = -lr * gradients / (np.sqrt(v) + self.epsilon)
            
        elif self.strategy == LearningStrategy.ADADELTA:
            v = 0.95 * v + 0.05 * (gradients ** 2)
            updates = -np.sqrt((self.velocities.get(param_id + '_delta', np.zeros_like(params)) + self.epsilon) / 
                              (v + self.epsilon)) * gradients
            self.velocities[param_id + '_delta'] = 0.95 * self.velocities.get(param_id + '_delta', np.zeros_like(params)) + \
                                                   0.05 * (updates ** 2)
            
        elif self.strategy == LearningStrategy.NADAM:
            m = self.beta1 * m + (1 - self.beta1) * gradients
            v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
            
            # Nesterov momentum
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            updates = -lr * (self.beta1 * m_hat + (1 - self.beta1) * gradients) / (np.sqrt(v_hat) + self.epsilon)
            
        elif self.strategy == LearningStrategy.ADAMW:
            m = self.beta1 * m + (1 - self.beta1) * gradients
            v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
            
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            # Decoupled weight decay
            updates = -lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params)
        
        else:
            updates = -lr * gradients
        
        # Update stored momentums and velocities
        self.momentums[param_id] = m
        self.velocities[param_id] = v
        
        return params + updates
    
    def record_loss(self, loss: float):
        """Record loss value for adaptive learning rate adjustment."""
        self.loss_history.append(loss)
        
        # Analyze performance trend
        if len(self.loss_history) >= 5:
            recent = list(self.loss_history)[-5:]
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            self.performance_trend.append(trend)
    
    def adapt_curriculum(self, current_performance: float, target_performance: float):
        """
        Adjust curriculum difficulty based on performance.
        
        Args:
            current_performance: Current model performance metric
            target_performance: Target performance level
        """
        performance_ratio = current_performance / target_performance
        
        if performance_ratio > 0.95:
            # Increase difficulty
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
            self.curriculum_stage += 1
        elif performance_ratio < 0.7:
            # Decrease difficulty
            self.difficulty_level = max(0.0, self.difficulty_level - 0.05)
    
    def meta_learn(self, task_gradients: List[np.ndarray], task_losses: List[float]):
        """
        Perform meta-learning update across multiple tasks.
        
        Args:
            task_gradients: List of gradients from different tasks
            task_losses: List of losses from different tasks
        """
        if len(task_gradients) < 2:
            return
        
        # Compute meta-gradients
        avg_gradient = np.mean(task_gradients, axis=0)
        gradient_variance = np.var(task_gradients, axis=0)
        
        # Adapt learning rate based on task similarity
        if np.mean(gradient_variance) < 0.1:
            # Tasks are similar, increase adaptation rate
            self.adaptation_rate = min(1.0, self.adaptation_rate * 1.1)
        else:
            # Tasks are diverse, decrease adaptation rate
            self.adaptation_rate = max(0.01, self.adaptation_rate * 0.9)
        
        # Store meta-gradients for future adaptation
        self.meta_gradients['avg'] = avg_gradient
        self.meta_gradients['variance'] = gradient_variance
    
    def get_optimization_state(self) -> Dict:
        """Return current optimization state."""
        return {
            'strategy': self.strategy.value,
            'current_lr': self.current_lr,
            'step_count': self.step_count,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else None,
            'loss_trend': self.performance_trend[-1] if self.performance_trend else None,
            'curriculum_stage': self.curriculum_stage,
            'difficulty_level': self.difficulty_level,
            'adaptation_rate': self.adaptation_rate
        }
    
    def reset_state(self, keep_history: bool = False):
        """Reset optimizer state while optionally keeping history."""
        self.momentums = {}
        self.velocities = {}
        self.step_count = 0
        self.current_lr = self.base_lr
        
        if not keep_history:
            self.loss_history.clear()
            self.performance_trend.clear()
            self.lr_schedule.clear()
    
    def save_checkpoint(self, filepath: str):
        """Save optimizer checkpoint."""
        checkpoint_data = {
            'strategy': self.strategy.value,
            'base_lr': self.base_lr,
            'current_lr': self.current_lr,
            'step_count': self.step_count,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'gradient_clip': self.gradient_clip,
            'momentums': {k: v.tolist() for k, v in self.momentums.items()},
            'velocities': {k: v.tolist() for k, v in self.velocities.items()},
            'loss_history': list(self.loss_history),
            'curriculum_stage': self.curriculum_stage,
            'difficulty_level': self.difficulty_level
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f)
    
    def load_checkpoint(self, filepath: str):
        """Load optimizer checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.strategy = LearningStrategy(checkpoint_data['strategy'])
        self.base_lr = checkpoint_data['base_lr']
        self.current_lr = checkpoint_data['current_lr']
        self.step_count = checkpoint_data['step_count']
        self.beta1 = checkpoint_data['beta1']
        self.beta2 = checkpoint_data['beta2']
        self.epsilon = checkpoint_data['epsilon']
        self.weight_decay = checkpoint_data['weight_decay']
        self.gradient_clip = checkpoint_data['gradient_clip']
        self.momentums = {k: np.array(v) for k, v in checkpoint_data['momentums'].items()}
        self.velocities = {k: np.array(v) for k, v in checkpoint_data['velocities'].items()}
        self.loss_history = deque(checkpoint_data['loss_history'], maxlen=100)
        self.curriculum_stage = checkpoint_data['curriculum_stage']
        self.difficulty_level = checkpoint_data['difficulty_level']
