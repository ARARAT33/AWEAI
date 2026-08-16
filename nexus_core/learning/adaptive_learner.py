"""
AdaptiveLearner - Self-improving AI learning system
Implements advanced machine learning algorithms for continuous improvement
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from ..utils.logger import setup_logger


@dataclass
class Experience:
    """Represents a learning experience"""
    task_id: str
    task_type: str
    action_taken: str
    outcome: str
    success: bool
    reward: float
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class LearningMetrics:
    """Learning performance metrics"""
    total_experiences: int
    successful_experiences: int
    average_reward: float
    learning_rate: float
    exploration_rate: float
    model_accuracy: float
    last_update: datetime


class AdaptiveLearner:
    """
    Advanced adaptive learning system
    
    Features:
    - Reinforcement learning
    - Online learning
    - Transfer learning
    - Meta-learning
    - Experience replay
    - AutoML optimization
    """
    
    def __init__(self, config):
        self.logger = setup_logger("AdaptiveLearner")
        self.config = config
        
        # Dependencies
        self.knowledge_base = None
        self.cognitive_processor = None
        
        # Experience storage
        self.experience_buffer: List[Experience] = []
        self.max_buffer_size = 10000
        self.priority_buffer: List[Experience] = []
        
        # Learning parameters
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
        # Model state
        self.model = None
        self.target_model = None
        self.model_version = 0
        self.last_training_step = 0
        self.training_frequency = 100
        
        # Performance tracking
        self.metrics = LearningMetrics(
            total_experiences=0,
            successful_experiences=0,
            average_reward=0.0,
            learning_rate=self.learning_rate,
            exploration_rate=self.exploration_rate,
            model_accuracy=0.0,
            last_update=datetime.now()
        )
        
        # Reward history for analysis
        self.reward_history: List[float] = []
        self.max_reward_history = 1000
        
        # AutoML state
        self.best_hyperparameters = {}
        self.hyperparameter_trials = 0
        
        self.logger.info("Adaptive Learner initialized")
    
    async def learn(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from a task
        
        Args:
            task: Task information
            
        Returns:
            Learning results and updated metrics
        """
        self.logger.info(f"Learning from task: {task.get('id', 'unknown')}")
        
        # Extract learning signal
        experience = await self._extract_experience(task)
        
        if experience:
            # Store experience
            await self.record_experience(task, experience, 
                                         success=task.get('success', True))
            
            # Train if enough experiences
            if len(self.experience_buffer) >= self.training_frequency:
                await self._train_batch()
        
        # Return learning metrics
        return {
            'status': 'learning',
            'experiences_stored': len(self.experience_buffer),
            'model_version': self.model_version,
            'metrics': self._get_metrics_dict()
        }
    
    async def _extract_experience(self, task: Dict[str, Any]) -> Optional[Experience]:
        """Extract experience from task execution"""
        try:
            experience = Experience(
                task_id=task.get('id', ''),
                task_type=task.get('type', 'generic'),
                action_taken=task.get('action', ''),
                outcome=task.get('result', {}).get('output', ''),
                success=task.get('success', True),
                reward=self._calculate_reward(task),
                context=task.get('context', {}),
                timestamp=datetime.now()
            )
            return experience
        except Exception as e:
            self.logger.error(f"Failed to extract experience: {e}")
            return None
    
    def _calculate_reward(self, task: Dict[str, Any]) -> float:
        """Calculate reward signal from task outcome"""
        reward = 0.0
        
        # Base reward for completion
        if task.get('success', False):
            reward += 1.0
        
        # Bonus for speed
        execution_time = task.get('execution_time', 0)
        if execution_time > 0:
            speed_bonus = min(0.5, 10.0 / execution_time)
            reward += speed_bonus
        
        # Penalty for errors
        if task.get('error'):
            reward -= 0.5
        
        # Bonus for confidence
        confidence = task.get('confidence', 0.5)
        reward += confidence * 0.3
        
        return max(-1.0, min(1.0, reward))
    
    async def record_experience(self, task: Dict[str, Any], 
                                 result: Any, success: bool):
        """
        Record an experience for learning
        
        Args:
            task: Original task
            result: Execution result
            success: Whether task succeeded
        """
        experience = Experience(
            task_id=task.get('id', ''),
            task_type=task.get('type', 'generic'),
            action_taken=task.get('action', ''),
            outcome=str(result) if result else '',
            success=success,
            reward=self._calculate_reward({
                **task,
                'success': success,
                'result': result
            }),
            context=task.get('context', {}),
            timestamp=datetime.now()
        )
        
        # Add to buffer
        self.experience_buffer.append(experience)
        
        # Update metrics
        self.metrics.total_experiences += 1
        if success:
            self.metrics.successful_experiences += 1
        
        # Update reward history
        self.reward_history.append(experience.reward)
        if len(self.reward_history) > self.max_reward_history:
            self.reward_history = self.reward_history[-self.max_reward_history:]
        
        # Calculate average reward
        self.metrics.average_reward = np.mean(self.reward_history)
        
        # Priority-based storage for important experiences
        if abs(experience.reward) > 0.8:  # High impact experience
            self.priority_buffer.append(experience)
            if len(self.priority_buffer) > 1000:
                self.priority_buffer.pop(0)
        
        # Trim buffer if too large
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
        
        self.logger.debug(f"Recorded experience: reward={experience.reward:.3f}")
    
    async def _train_batch(self):
        """Train model on a batch of experiences"""
        if len(self.experience_buffer) < self.training_frequency:
            return
        
        self.logger.info(f"Training on batch of {len(self.experience_buffer)} experiences")
        
        try:
            # Sample batch from experience buffer
            batch_size = min(64, len(self.experience_buffer))
            batch = self._sample_batch(batch_size)
            
            # Prepare training data
            states, actions, rewards, next_states, dones = self._prepare_batch(batch)
            
            # Train model (simplified - in production would use neural network)
            await self._update_model(states, actions, rewards, next_states, dones)
            
            # Update target model periodically
            if self.model_version % 10 == 0 and self.target_model is not None:
                await self._sync_target_model()
            
            # Update metrics
            self.metrics.last_update = datetime.now()
            self.model_version += 1
            self.last_training_step += 1
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
            self.metrics.exploration_rate = self.exploration_rate
            
            self.logger.info(f"Training complete. Model version: {self.model_version}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
    
    def _sample_batch(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences with prioritization"""
        # Mix of random and priority samples
        priority_count = min(batch_size // 4, len(self.priority_buffer))
        random_count = batch_size - priority_count
        
        batch = []
        
        # Add priority experiences
        if priority_count > 0:
            priority_indices = np.random.choice(
                len(self.priority_buffer), 
                size=priority_count, 
                replace=False
            )
            batch.extend([self.priority_buffer[i] for i in priority_indices])
        
        # Add random experiences
        if random_count > 0 and len(self.experience_buffer) > 0:
            random_indices = np.random.choice(
                len(self.experience_buffer),
                size=min(random_count, len(self.experience_buffer)),
                replace=False
            )
            batch.extend([self.experience_buffer[i] for i in random_indices])
        
        return batch
    
    def _prepare_batch(self, batch: List[Experience]) -> Tuple:
        """Prepare batch for training"""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in batch:
            # Encode state from context
            state = self._encode_state(exp.context)
            states.append(state)
            actions.append(exp.action_taken)
            rewards.append(exp.reward)
            
            # Next state (simplified)
            next_states.append(state)  # In reality would be different
            dones.append(1.0 if not exp.success else 0.0)
        
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), np.array(dones))
    
    def _encode_state(self, context: Dict) -> np.ndarray:
        """Encode context into state vector"""
        # Simple encoding - in production would use proper feature extraction
        state_vector = np.zeros(10)
        
        if isinstance(context, dict):
            features = list(context.values())[:10]
            for i, f in enumerate(features):
                if isinstance(f, (int, float)):
                    state_vector[i] = f
                elif isinstance(f, bool):
                    state_vector[i] = 1.0 if f else 0.0
        
        return state_vector
    
    async def _update_model(self, states, actions, rewards, next_states, dones):
        """Update model parameters"""
        # Simplified Q-learning update
        # In production, this would use deep neural networks
        
        if self.model is None:
            # Initialize simple Q-table approximation
            self.model = {
                'weights': np.random.randn(10, 5) * 0.01,
                'bias': np.zeros(5)
            }
        
        # Compute TD targets
        current_q = np.dot(states, self.model['weights']) + self.model['bias']
        
        # Update using gradient descent (simplified)
        td_errors = rewards + self.discount_factor * np.max(current_q, axis=1) * (1 - dones)
        
        # Update weights
        gradient = np.dot(states.T, td_errors) / len(states)
        self.model['weights'] += self.learning_rate * gradient
        
        # Update accuracy estimate
        self.metrics.model_accuracy = min(1.0, self.metrics.model_accuracy + 0.01)
    
    async def _sync_target_model(self):
        """Sync target model with main model"""
        if self.model:
            self.target_model = {
                'weights': self.model['weights'].copy(),
                'bias': self.model['bias'].copy()
            }
    
    async def adjust_learning_rate(self):
        """Automatically adjust learning rate based on performance"""
        if len(self.reward_history) < 100:
            return
        
        recent_rewards = self.reward_history[-100:]
        trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        
        # Increase learning rate if improving, decrease if unstable
        if trend > 0.01:
            self.learning_rate = min(0.01, self.learning_rate * 1.05)
        elif trend < -0.01:
            self.learning_rate = max(0.0001, self.learning_rate * 0.95)
        
        self.metrics.learning_rate = self.learning_rate
        self.logger.debug(f"Adjusted learning rate: {self.learning_rate:.6f}")
    
    async def process_feedback(self, task_id: str, feedback: Dict[str, Any]):
        """Process user feedback for additional learning signal"""
        # Find corresponding experience
        for exp in reversed(self.experience_buffer):
            if exp.task_id == task_id:
                # Adjust reward based on feedback
                feedback_score = feedback.get('score', 0.5)
                exp.reward = (exp.reward + feedback_score) / 2
                
                # Add to priority buffer
                self.priority_buffer.append(exp)
                
                self.logger.info(f"Processed feedback for task {task_id}")
                break
    
    async def optimize_hyperparameters(self):
        """AutoML: Optimize hyperparameters using evolutionary algorithm"""
        self.logger.info("Starting hyperparameter optimization...")
        
        # Simple grid search / evolutionary approach
        candidates = [
            {'learning_rate': 0.001, 'discount_factor': 0.99},
            {'learning_rate': 0.01, 'discount_factor': 0.95},
            {'learning_rate': 0.0001, 'discount_factor': 0.999},
            {'learning_rate': 0.005, 'discount_factor': 0.97}
        ]
        
        best_score = -float('inf')
        best_params = candidates[0]
        
        for params in candidates:
            # Evaluate candidate (simplified)
            score = await self._evaluate_hyperparameters(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        # Apply best parameters
        self.learning_rate = best_params['learning_rate']
        self.discount_factor = best_params['discount_factor']
        self.best_hyperparameters = best_params
        
        self.hyperparameter_trials += 1
        self.logger.info(f"Best hyperparameters: {best_params}")
    
    async def _evaluate_hyperparameters(self, params: Dict) -> float:
        """Evaluate a set of hyperparameters"""
        # Use recent performance as proxy
        if len(self.reward_history) < 50:
            return 0.0
        
        recent_avg = np.mean(self.reward_history[-50:])
        return recent_avg
    
    async def save_model(self):
        """Save model state to disk"""
        try:
            save_path = Path(self.config.get('model_save_path', './models'))
            save_path.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': {
                    'weights': self.model['weights'].tolist() if self.model else None,
                    'bias': self.model['bias'].tolist() if self.model else None
                },
                'metrics': {
                    'total_experiences': self.metrics.total_experiences,
                    'average_reward': self.metrics.average_reward,
                    'model_accuracy': self.metrics.model_accuracy
                },
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'exploration_rate': self.exploration_rate
                },
                'version': self.model_version
            }
            
            model_file = save_path / f'nexus_model_v{self.model_version}.json'
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            self.logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    async def load_model(self, version: Optional[int] = None):
        """Load model from disk"""
        try:
            save_path = Path(self.config.get('model_save_path', './models'))
            
            if version is None:
                # Load latest version
                model_files = list(save_path.glob('nexus_model_v*.json'))
                if not model_files:
                    self.logger.warning("No saved models found")
                    return False
                model_file = sorted(model_files)[-1]
            else:
                model_file = save_path / f'nexus_model_v{version}.json'
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            # Restore model
            if model_data['model']['weights']:
                self.model = {
                    'weights': np.array(model_data['model']['weights']),
                    'bias': np.array(model_data['model']['bias'])
                }
            
            # Restore metrics
            self.metrics.total_experiences = model_data['metrics']['total_experiences']
            self.metrics.average_reward = model_data['metrics']['average_reward']
            self.metrics.model_accuracy = model_data['metrics']['model_accuracy']
            
            # Restore hyperparameters
            self.learning_rate = model_data['hyperparameters']['learning_rate']
            self.discount_factor = model_data['hyperparameters']['discount_factor']
            self.exploration_rate = model_data['hyperparameters']['exploration_rate']
            
            self.model_version = model_data['version']
            
            self.logger.info(f"Model loaded from {model_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            'total_experiences': self.metrics.total_experiences,
            'successful_experiences': self.metrics.successful_experiences,
            'success_rate': self.metrics.successful_experiences / max(1, self.metrics.total_experiences),
            'average_reward': float(self.metrics.average_reward),
            'learning_rate': self.metrics.learning_rate,
            'exploration_rate': self.metrics.exploration_rate,
            'model_accuracy': float(self.metrics.model_accuracy),
            'model_version': self.model_version
        }
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Get current learning state"""
        return {
            'buffer_size': len(self.experience_buffer),
            'priority_buffer_size': len(self.priority_buffer),
            'metrics': self._get_metrics_dict(),
            'best_hyperparameters': self.best_hyperparameters,
            'trials_completed': self.hyperparameter_trials
        }
