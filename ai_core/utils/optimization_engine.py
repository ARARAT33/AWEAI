"""
Optimization Engine - Advanced Gradient Optimization and Hyperparameter Tuning
Implements state-of-the-art optimization algorithms with automatic tuning,
gradient analysis, and convergence monitoring.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import deque
import json


class GradientOptimizer:
    """
    Advanced gradient optimization with multiple strategies:
    - Adaptive moment estimation
    - Gradient clipping and scaling
    - Learning rate warmup and decay
    - Second-order optimization approximations
    - Gradient noise injection
    """
    
    def __init__(
        self,
        optimizer_type: str = 'adam',
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        gradient_clip_value: float = 1.0,
        gradient_clip_norm: float = None
    ):
        self.optimizer_type = optimizer_type
        self.base_lr = learning_rate
        self.current_lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_norm = gradient_clip_norm
        
        # Optimizer state
        self.momentums = {}
        self.velocities = {}
        self.second_moments = {}
        self.step_count = 0
        
        # Gradient statistics
        self.gradient_history = deque(maxlen=100)
        self.gradient_norms = []
        
    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Apply gradient clipping."""
        if self.gradient_clip_norm is not None:
            norm = np.linalg.norm(gradients)
            if norm > self.gradient_clip_norm:
                gradients = gradients * (self.gradient_clip_norm / norm)
        
        if self.gradient_clip_value is not None:
            gradients = np.clip(gradients, -self.gradient_clip_value, self.gradient_clip_value)
        
        return gradients
    
    def compute_gradient_statistics(self, gradients: np.ndarray) -> Dict[str, float]:
        """Compute statistics about gradients for monitoring."""
        stats = {
            'mean': float(np.mean(gradients)),
            'std': float(np.std(gradients)),
            'min': float(np.min(gradients)),
            'max': float(np.max(gradients)),
            'norm': float(np.linalg.norm(gradients)),
            'sparsity': float(np.mean(np.abs(gradients) < 1e-6))
        }
        
        self.gradient_history.append(stats['norm'])
        self.gradient_norms.append(stats['norm'])
        
        return stats
    
    def optimize_step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: str = "default"
    ) -> np.ndarray:
        """Perform one optimization step."""
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        # Compute gradient statistics
        grad_stats = self.compute_gradient_statistics(gradients)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        self.step_count += 1
        
        # Initialize state if needed
        if param_id not in self.momentums:
            self.momentums[param_id] = np.zeros_like(params)
            self.velocities[param_id] = np.zeros_like(params)
            self.second_moments[param_id] = np.zeros_like(params)
        
        m = self.momentums[param_id]
        v = self.velocities[param_id]
        s = self.second_moments[param_id]
        
        # Apply optimization algorithm
        if self.optimizer_type == 'sgd':
            update = -self.current_lr * gradients
            
        elif self.optimizer_type == 'momentum':
            m = self.beta1 * m - self.current_lr * gradients
            update = m
            
        elif self.optimizer_type == 'nesterov':
            m_prev = m.copy()
            m = self.beta1 * m - self.current_lr * gradients
            update = -self.beta1 * m_prev + (1 + self.beta1) * m
            
        elif self.optimizer_type == 'adam':
            m = self.beta1 * m + (1 - self.beta1) * gradients
            v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
            
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            update = -self.current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
        elif self.optimizer_type == 'adamw':
            m = self.beta1 * m + (1 - self.beta1) * gradients
            v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
            
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            # Decoupled weight decay
            update = -self.current_lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params)
            
        elif self.optimizer_type == 'rmsprop':
            v = 0.9 * v + 0.1 * (gradients ** 2)
            update = -self.current_lr * gradients / (np.sqrt(v) + self.epsilon)
            
        elif self.optimizer_type == 'adagrad':
            s = s + gradients ** 2
            update = -self.current_lr * gradients / (np.sqrt(s) + self.epsilon)
            
        elif self.optimizer_type == 'adadelta':
            rho = 0.95
            s = rho * s + (1 - rho) * gradients ** 2
            delta_x = np.sqrt((self.second_moments.get(param_id + '_delta', np.zeros_like(params)) + self.epsilon) / 
                             (s + self.epsilon)) * gradients
            update = delta_x
            self.second_moments[param_id + '_delta'] = rho * self.second_moments.get(param_id + '_delta', np.zeros_like(params)) + \
                                                       (1 - rho) * (delta_x ** 2)
            
        else:
            update = -self.current_lr * gradients
        
        # Update stored state
        self.momentums[param_id] = m
        self.velocities[param_id] = v
        self.second_moments[param_id] = s
        
        return params + update
    
    def set_learning_rate(self, lr: float):
        """Set current learning rate."""
        self.current_lr = lr
    
    def get_learning_rate_schedule(
        self,
        schedule_type: str = 'cosine',
        total_steps: int = 10000,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6
    ) -> float:
        """Get learning rate based on schedule."""
        if self.step_count < warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count + 1) / warmup_steps
        else:
            progress = (self.step_count - warmup_steps) / (total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            
            if schedule_type == 'cosine':
                lr = min_lr + (self.base_lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2
            elif schedule_type == 'linear':
                lr = self.base_lr * (1 - progress)
            elif schedule_type == 'exponential':
                lr = self.base_lr * np.exp(-10 * progress)
            elif schedule_type == 'step':
                step_size = total_steps // 3
                lr = self.base_lr * (0.1 ** (self.step_count // step_size))
            else:
                lr = self.base_lr
        
        self.current_lr = lr
        return lr
    
    def add_gradient_noise(self, gradients: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """Add noise to gradients for regularization."""
        noise = np.random.randn(*gradients.shape) * noise_scale
        return gradients + noise
    
    def get_optimizer_state(self) -> Dict[str, Any]:
        """Get current optimizer state."""
        return {
            'optimizer_type': self.optimizer_type,
            'step_count': self.step_count,
            'current_lr': self.current_lr,
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'momentum_keys': list(self.momentums.keys()),
            'velocity_keys': list(self.velocities.keys())
        }
    
    def reset_state(self):
        """Reset optimizer state."""
        self.momentums = {}
        self.velocities = {}
        self.second_moments = {}
        self.step_count = 0
        self.current_lr = self.base_lr
        self.gradient_history.clear()
        self.gradient_norms.clear()


class OptimizationEngine:
    """
    Complete optimization engine with hyperparameter tuning,
    early stopping, and convergence monitoring.
    """
    
    def __init__(
        self,
        optimizer_cls: type = GradientOptimizer,
        optimizer_config: Dict = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        convergence_window: int = 20
    ):
        self.optimizer = optimizer_cls(**(optimizer_config or {}))
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.convergence_window = convergence_window
        
        # Training state
        self.loss_history = deque(maxlen=100)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.convergence_scores = []
        
        # Hyperparameter search state
        self.hp_search_history = []
        self.best_hyperparameters = None
        
    def check_early_stopping(self, current_loss: float) -> bool:
        """Check if training should stop early."""
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        return self.patience_counter >= self.early_stopping_patience
    
    def check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.loss_history) < self.convergence_window:
            return False
        
        recent_losses = list(self.loss_history)[-self.convergence_window:]
        loss_variance = np.var(recent_losses)
        
        converged = loss_variance < self.early_stopping_min_delta
        self.convergence_scores.append(loss_variance)
        
        return converged
    
    def hyperparameter_search(
        self,
        model,
        train_fn: Callable,
        param_grid: Dict[str, List],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_folds: int = 3,
        metric_fn: Callable = None
    ) -> Dict[str, Any]:
        """
        Perform grid search over hyperparameters.
        
        Args:
            model: Model class to optimize
            train_fn: Training function
            param_grid: Dictionary of parameter names to lists of values
            X_train: Training data
            y_train: Training labels
            n_folds: Number of cross-validation folds
            metric_fn: Metric function (defaults to negative loss)
            
        Returns:
            Best hyperparameters and score
        """
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        best_score = float('-inf')
        best_params = {}
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            # Cross-validation
            fold_scores = []
            indices = np.random.permutation(len(X_train))
            fold_size = len(X_train) // n_folds
            
            for fold in range(n_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_folds - 1 else len(X_train)
                
                val_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                X_fold_train = X_train[train_indices]
                y_fold_train = y_train[train_indices]
                X_fold_val = X_train[val_indices]
                y_fold_val = y_train[val_indices]
                
                # Train model with these hyperparameters
                try:
                    score = train_fn(X_fold_train, y_fold_train, X_fold_val, y_fold_val, params)
                    fold_scores.append(score)
                except Exception as e:
                    fold_scores.append(float('-inf'))
            
            avg_score = np.mean(fold_scores)
            
            self.hp_search_history.append({
                'params': params,
                'score': avg_score,
                'fold_scores': fold_scores
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        self.best_hyperparameters = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.hp_search_history
        }
    
    def optimize_with_line_search(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        direction: np.ndarray,
        param_id: str = "default",
        max_iterations: int = 20
    ) -> Tuple[np.ndarray, float]:
        """
        Perform line search to find optimal step size.
        
        Args:
            params: Current parameters
            gradients: Current gradients
            direction: Search direction
            param_id: Parameter identifier
            max_iterations: Maximum line search iterations
            
        Returns:
            Updated parameters and optimal step size
        """
        # Backtracking line search
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        
        # Simple quadratic approximation for initial loss
        f0 = np.sum(params ** 2)  # Placeholder
        
        for _ in range(max_iterations):
            new_params = params + alpha * direction
            
            # Approximate new loss (in practice, this would be actual loss computation)
            f_new = np.sum(new_params ** 2)
            
            # Armijo condition
            if f_new <= f0 + c * alpha * np.sum(gradients * direction):
                break
            
            alpha *= rho
        
        # Apply update with optimal step size
        updated_params = self.optimizer.optimize_step(params, gradients * alpha, param_id)
        
        return updated_params, alpha
    
    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive training diagnostics."""
        diagnostics = {
            'loss_history': list(self.loss_history),
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'convergence_status': self.check_convergence(),
            'early_stopping_status': self.check_early_stopping(self.loss_history[-1]) if self.loss_history else False,
            'optimizer_state': self.optimizer.get_optimizer_state(),
            'convergence_scores': self.convergence_scores[-10:] if self.convergence_scores else []
        }
        
        if len(self.loss_history) > 1:
            losses = list(self.loss_history)
            diagnostics['loss_trend'] = np.polyfit(range(len(losses)), losses, 1)[0]
            diagnostics['loss_improvement'] = losses[0] - losses[-1]
        
        return diagnostics
    
    def save_checkpoint(self, filepath: str):
        """Save optimization checkpoint."""
        checkpoint = {
            'optimizer_state': self.optimizer.get_optimizer_state(),
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'loss_history': list(self.loss_history),
            'best_hyperparameters': self.best_hyperparameters
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self, filepath: str):
        """Load optimization checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.best_loss = checkpoint['best_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.loss_history = deque(checkpoint['loss_history'], maxlen=100)
        self.best_hyperparameters = checkpoint['best_hyperparameters']
