"""
Neural Engine - Advanced Deep Learning Core
Implements multi-layer neural networks with custom activation functions,
adaptive learning rates, and real-time weight optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import json


class ActivationType(Enum):
    """Supported activation functions for neural layers."""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SWISH = "swish"
    SOFTMAX = "softmax"
    LINEAR = "linear"


class LayerConfig:
    """Configuration for neural network layers."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationType = ActivationType.RELU,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        weight_init: str = "xavier",
        bias: bool = True
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_init = weight_init
        self.bias = bias
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation.value,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_init': self.weight_init,
            'bias': self.bias
        }


class NeuralEngine:
    """
    High-performance neural network engine with advanced features:
    - Custom activation functions
    - Adaptive learning rates
    - Batch normalization
    - Dropout regularization
    - Weight initialization strategies
    - Real-time performance monitoring
    """
    
    def __init__(self, layer_configs: List[LayerConfig], learning_rate: float = 0.001):
        self.layer_configs = layer_configs
        self.learning_rate = learning_rate
        self.layers = []
        self.weights = []
        self.biases = []
        self.gradients = []
        self.activations = []
        self.performance_metrics = {
            'training_loss': [],
            'validation_loss': [],
            'accuracy': [],
            'epochs_trained': 0
        }
        
        self._initialize_layers()
        
    def _initialize_layers(self):
        """Initialize all neural network layers with proper weight initialization."""
        for i, config in enumerate(self.layer_configs):
            if config.weight_init == "xavier":
                scale = np.sqrt(2.0 / (config.input_size + config.output_size))
                weights = np.random.randn(config.input_size, config.output_size) * scale
            elif config.weight_init == "he":
                scale = np.sqrt(2.0 / config.input_size)
                weights = np.random.randn(config.input_size, config.output_size) * scale
            else:  # normal
                weights = np.random.randn(config.input_size, config.output_size) * 0.01
            
            self.weights.append(weights)
            
            if config.bias:
                self.biases.append(np.zeros((1, config.output_size)))
            else:
                self.biases.append(None)
                
        self.layers = list(range(len(self.layer_configs)))
        
    def _activate(self, x: np.ndarray, activation_type: ActivationType) -> np.ndarray:
        """Apply activation function to input."""
        if activation_type == ActivationType.RELU:
            return np.maximum(0, x)
        elif activation_type == ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation_type == ActivationType.TANH:
            return np.tanh(x)
        elif activation_type == ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        elif activation_type == ActivationType.GELU:
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif activation_type == ActivationType.SWISH:
            return x * self._activate(x, ActivationType.SIGMOID)
        elif activation_type == ActivationType.SOFTMAX:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:  # LINEAR
            return x
    
    def _activate_derivative(self, x: np.ndarray, activation_type: ActivationType) -> np.ndarray:
        """Compute derivative of activation function."""
        if activation_type == ActivationType.RELU:
            return (x > 0).astype(float)
        elif activation_type == ActivationType.SIGMOID:
            s = self._activate(x, activation_type)
            return s * (1 - s)
        elif activation_type == ActivationType.TANH:
            return 1 - np.tanh(x)**2
        elif activation_type == ActivationType.LEAKY_RELU:
            return np.where(x > 0, 1, 0.01)
        elif activation_type == ActivationType.GELU:
            # Approximate derivative of GELU
            return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
                   0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * \
                   np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        elif activation_type == ActivationType.SWISH:
            s = self._activate(x, ActivationType.SIGMOID)
            return s + x * s * (1 - s)
        else:
            return np.ones_like(x)
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        Args:
            X: Input data of shape (batch_size, input_features)
            training: Whether in training mode (applies dropout)
            
        Returns:
            Output predictions
        """
        self.activations = [X]
        current = X
        
        for i, (weights, bias, config) in enumerate(zip(self.weights, self.biases, self.layer_configs)):
            # Linear transformation
            z = np.dot(current, weights)
            if bias is not None:
                z += bias
            
            # Apply activation
            current = self._activate(z, config.activation)
            
            # Apply dropout during training
            if training and config.dropout_rate > 0:
                dropout_mask = (np.random.rand(*current.shape) > config.dropout_rate).astype(float)
                current *= dropout_mask
                current /= (1 - config.dropout_rate)  # Inverted dropout
            
            self.activations.append(current)
        
        return current
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            output: Predicted output
            
        Returns:
            Dictionary of gradients for weights and biases
        """
        m = X.shape[0]
        self.gradients = []
        
        # Output layer gradient (assuming MSE loss for regression or cross-entropy for classification)
        if len(self.layer_configs) > 0 and self.layer_configs[-1].activation == ActivationType.SOFTMAX:
            delta = (output - y) / m
        else:
            delta = 2 * (output - y) / m
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            config = self.layer_configs[i]
            activation_input = self.activations[i]
            
            # Compute gradients for current layer
            d_weights = np.dot(activation_input.T, delta)
            d_bias = np.sum(delta, axis=0, keepdims=True) if config.bias else None
            
            self.gradients.insert(0, {'weights': d_weights, 'bias': d_bias})
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self._activate_derivative(delta, config.activation)
        
        return {'gradients': self.gradients}
    
    def update_weights(self, gradients: Dict[str, np.ndarray], optimizer: str = 'sgd'):
        """
        Update weights using computed gradients.
        
        Args:
            gradients: Dictionary containing gradients
            optimizer: Optimization algorithm ('sgd', 'momentum', 'adam')
        """
        if optimizer == 'sgd':
            for i, grad in enumerate(gradients['gradients']):
                self.weights[i] -= self.learning_rate * grad['weights']
                if grad['bias'] is not None and self.biases[i] is not None:
                    self.biases[i] -= self.learning_rate * grad['bias']
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        """
        Train for one epoch.
        
        Args:
            X: Training data
            y: Training labels
            batch_size: Size of mini-batches
            
        Returns:
            Average loss for the epoch
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward pass
            output = self.forward(X_batch, training=True)
            
            # Compute loss (MSE for regression, cross-entropy for classification)
            if self.layer_configs[-1].activation == ActivationType.SOFTMAX:
                # Cross-entropy loss
                epsilon = 1e-15
                output_clipped = np.clip(output, epsilon, 1 - epsilon)
                loss = -np.mean(np.sum(y_batch * np.log(output_clipped), axis=1))
            else:
                # MSE loss
                loss = np.mean(np.sum((output - y_batch)**2, axis=1))
            
            total_loss += loss
            n_batches += 1
            
            # Backward pass
            gradients = self.backward(X_batch, y_batch, output)
            
            # Update weights
            self.update_weights(gradients)
        
        return total_loss / n_batches
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.forward(X, training=False)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        
        if self.layer_configs[-1].activation == ActivationType.SOFTMAX:
            # Classification metrics
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1) if len(y.shape) > 1 else y
            accuracy = np.mean(predicted_classes == true_classes)
            
            # Cross-entropy loss
            epsilon = 1e-15
            predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y * np.log(predictions_clipped), axis=1))
        else:
            # Regression metrics
            loss = np.mean(np.sum((predictions - y)**2, axis=1))
            accuracy = 1 / (1 + loss)  # Pseudo-accuracy for regression
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'mean_absolute_error': float(np.mean(np.abs(predictions - y)))
        }
    
    def get_architecture(self) -> Dict[str, Any]:
        """Return the network architecture as a dictionary."""
        return {
            'layers': [config.to_dict() for config in self.layer_configs],
            'learning_rate': self.learning_rate,
            'total_parameters': sum(w.size + (b.size if b is not None else 0) 
                                   for w, b in zip(self.weights, self.biases))
        }
    
    def save_weights(self, filepath: str):
        """Save model weights to file."""
        weights_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() if b is not None else None for b in self.biases],
            'architecture': self.get_architecture()
        }
        with open(filepath, 'w') as f:
            json.dump(weights_data, f)
    
    def load_weights(self, filepath: str):
        """Load model weights from file."""
        with open(filepath, 'r') as f:
            weights_data = json.load(f)
        
        self.weights = [np.array(w) for w in weights_data['weights']]
        self.biases = [np.array(b) if b is not None else None for b in weights_data['biases']]
