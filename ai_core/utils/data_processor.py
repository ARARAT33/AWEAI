"""
Data Processor - Advanced Data Pipeline and Feature Engineering
Implements comprehensive data preprocessing, augmentation, and feature extraction
with support for multiple data modalities and real-time processing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import json


class FeatureExtractor:
    """
    Advanced feature extraction with multiple techniques:
    - Statistical features
    - Time-series features
    - Text features
    - Image features
    - Dimensionality reduction
    """
    
    def __init__(
        self,
        feature_types: List[str] = None,
        normalize: bool = True,
        handle_missing: str = 'mean'
    ):
        self.feature_types = feature_types or ['statistical', 'temporal']
        self.normalize = normalize
        self.handle_missing = handle_missing
        
        # Fitted statistics
        self.mean = None
        self.std = None
        self.min_vals = None
        self.max_vals = None
        self.feature_names = []
        
    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """Extract statistical features from data."""
        features = []
        
        # Basic statistics
        features.append(np.mean(data, axis=-1))
        features.append(np.std(data, axis=-1))
        features.append(np.min(data, axis=-1))
        features.append(np.max(data, axis=-1))
        features.append(np.median(data, axis=-1))
        
        # Higher-order statistics
        if data.shape[-1] > 2:
            # Skewness approximation
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True) + 1e-9
            skewness = np.mean(((data - mean) / std) ** 3, axis=-1)
            features.append(skewness)
            
            # Kurtosis approximation
            kurtosis = np.mean(((data - mean) / std) ** 4, axis=-1) - 3
            features.append(kurtosis)
        
        return np.column_stack(features)
    
    def extract_temporal_features(self, data: np.ndarray) -> np.ndarray:
        """Extract time-series features."""
        features = []
        
        # Lag features
        if data.shape[-1] > 1:
            for lag in [1, 2, 3]:
                if data.shape[-1] > lag:
                    features.append(data[..., :-lag].mean(axis=-1))
        
        # Rolling statistics (window of 3)
        if data.shape[-1] >= 3:
            window = 3
            rolling_mean = np.array([
                np.mean(data[..., i:i+window], axis=-1) 
                for i in range(data.shape[-1] - window + 1)
            ]).T
            features.append(rolling_mean.mean(axis=-1))
            features.append(rolling_mean.std(axis=-1))
        
        # Trend detection
        if data.shape[-1] > 2:
            x = np.arange(data.shape[-1])
            trends = []
            for i in range(len(data)):
                if len(data.shape) == 1:
                    sample = data
                else:
                    sample = data[i]
                trend = np.polyfit(x, sample, 1)[0]
                trends.append(trend)
            features.append(np.array(trends))
        
        if features:
            return np.column_stack(features) if len(features) > 1 else np.array(features).reshape(-1, 1)
        return np.zeros((len(data), 1))
    
    def fit(self, data: np.ndarray):
        """Fit the feature extractor on training data."""
        if self.normalize:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-9
            self.min_vals = np.min(data, axis=0)
            self.max_vals = np.max(data, axis=0) + 1e-9
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data by extracting features."""
        all_features = []
        
        for feature_type in self.feature_types:
            if feature_type == 'statistical':
                features = self.extract_statistical_features(data)
            elif feature_type == 'temporal':
                features = self.extract_temporal_features(data)
            else:
                continue
            
            all_features.append(features)
        
        if all_features:
            extracted = np.hstack(all_features)
        else:
            extracted = data.copy()
        
        # Handle missing values
        if self.handle_missing == 'mean':
            extracted = np.nan_to_num(extracted, nan=0.0)
        elif self.handle_missing == 'zero':
            extracted = np.nan_to_num(extracted, nan=0.0)
        
        # Normalize
        if self.normalize and self.mean is not None:
            if extracted.shape[1] == self.mean.shape[0]:
                extracted = (extracted - self.mean) / self.std
        
        return extracted
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class DataProcessor:
    """
    Comprehensive data processing pipeline with:
    - Data cleaning and validation
    - Normalization and standardization
    - Data augmentation
    - Batch generation
    - Multi-modal data handling
    - Real-time preprocessing
    """
    
    def __init__(
        self,
        normalization: str = 'zscore',
        augmentation_enabled: bool = True,
        batch_size: int = 32,
        shuffle: bool = True,
        validation_split: float = 0.2
    ):
        self.normalization = normalization
        self.augmentation_enabled = augmentation_enabled
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        
        # Processing state
        self.fitted = False
        self.data_stats = {}
        self.class_weights = None
        self.label_encoders = {}
        
        # Augmentation parameters
        self.noise_std = 0.01
        self.scale_range = (0.9, 1.1)
        self.rotation_range = 10  # degrees
        
    def _validate_data(self, data: np.ndarray) -> bool:
        """Validate input data quality."""
        if data.size == 0:
            raise ValueError("Empty data array")
        
        if np.any(np.isnan(data)):
            print(f"Warning: Found {np.sum(np.isnan(data))} NaN values")
        
        if np.any(np.isinf(data)):
            print(f"Warning: Found {np.sum(np.isinf(data))} Inf values")
        
        return True
    
    def _normalize(self, data: np.ndarray, method: str = None) -> np.ndarray:
        """Normalize data using specified method."""
        method = method or self.normalization
        
        if method == 'zscore':
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True) + 1e-9
            return (data - mean) / std
        
        elif method == 'minmax':
            min_val = np.min(data, axis=0, keepdims=True)
            max_val = np.max(data, axis=0, keepdims=True) + 1e-9
            return (data - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            median = np.median(data, axis=0, keepdims=True)
            q75 = np.percentile(data, 75, axis=0, keepdims=True)
            q25 = np.percentile(data, 25, axis=0, keepdims=True)
            iqr = q75 - q25 + 1e-9
            return (data - median) / iqr
        
        elif method == 'l2':
            norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-9
            return data / norms
        
        return data
    
    def _augment_data(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply data augmentation techniques."""
        if not self.augmentation_enabled:
            return data, labels
        
        augmented_data = data.copy()
        
        # Add Gaussian noise
        noise = np.random.randn(*data.shape) * self.noise_std
        augmented_data = augmented_data + noise
        
        # Random scaling
        if len(data.shape) > 1:
            scale_factor = np.random.uniform(*self.scale_range)
            augmented_data = augmented_data * scale_factor
        
        # Random clipping (simulating dropout at data level)
        if np.random.random() < 0.3:
            mask = np.random.rand(*augmented_data.shape) > 0.1
            augmented_data = augmented_data * mask
        
        if labels is not None:
            return augmented_data, labels.copy()
        return augmented_data, None
    
    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        self.class_weights = {
            int(cls): total / (len(unique) * count) 
            for cls, count in zip(unique, counts)
        }
        
        return self.class_weights
    
    def encode_labels(self, labels: np.ndarray, encoding_type: str = 'onehot') -> np.ndarray:
        """Encode labels for training."""
        unique_labels = np.unique(labels)
        
        if encoding_type == 'onehot':
            n_classes = len(unique_labels)
            encoded = np.zeros((len(labels), n_classes))
            
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.label_encoders['onehot_map'] = label_map
            
            for i, label in enumerate(labels):
                encoded[i, label_map[label]] = 1
            
            return encoded
        
        elif encoding_type == 'integer':
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.label_encoders['int_map'] = label_map
            return np.array([label_map[label] for label in labels])
        
        return labels
    
    def create_train_val_split(
        self, 
        data: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        stratified: bool = True
    ) -> Union[Tuple, Dict]:
        """Split data into training and validation sets."""
        n_samples = len(data)
        n_val = int(n_samples * self.validation_split)
        
        indices = np.arange(n_samples)
        
        if stratified and labels is not None:
            # Stratified split
            unique_labels = np.unique(labels)
            train_indices = []
            val_indices = []
            
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                np.random.shuffle(label_indices)
                
                n_val_label = max(1, int(len(label_indices) * self.validation_split))
                val_indices.extend(label_indices[:n_val_label])
                train_indices.extend(label_indices[n_val_label:])
            
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
        else:
            if self.shuffle:
                np.random.shuffle(indices)
            
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]
        
        X_train = data[train_indices]
        X_val = data[val_indices]
        
        if labels is not None:
            y_train = labels[train_indices]
            y_val = labels[val_indices]
            return X_train, X_val, y_train, y_val
        
        return {'train': X_train, 'validation': X_val}
    
    def generate_batches(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        augment: bool = False,
        infinite: bool = False
    ):
        """Generate batches of data for training."""
        n_samples = len(data)
        indices = np.arange(n_samples)
        
        while True:
            if self.shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = data[batch_indices]
                
                if augment:
                    X_batch, _ = self._augment_data(X_batch)
                
                if labels is not None:
                    y_batch = labels[batch_indices]
                    yield X_batch, y_batch
                else:
                    yield X_batch
            
            if not infinite:
                break
    
    def process_pipeline(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        fit: bool = True,
        augment: bool = False
    ) -> Union[Tuple, np.ndarray]:
        """
        Complete data processing pipeline.
        
        Args:
            data: Input data
            labels: Optional labels
            fit: Whether to fit preprocessing parameters
            augment: Whether to apply augmentation
            
        Returns:
            Processed data (and labels if provided)
        """
        # Validate
        self._validate_data(data)
        
        # Compute statistics if fitting
        if fit:
            self.data_stats = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0),
                'shape': data.shape
            }
            self.fitted = True
        
        # Normalize
        processed = self._normalize(data)
        
        # Augment if requested
        if augment:
            processed, aug_labels = self._augment_data(processed, labels)
            if labels is not None:
                labels = aug_labels
        
        if labels is not None:
            return processed, labels
        return processed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return computed data statistics."""
        return {
            'fitted': self.fitted,
            'statistics': self.data_stats,
            'class_weights': self.class_weights,
            'normalization_method': self.normalization,
            'batch_size': self.batch_size
        }
    
    def save_preprocessing_state(self, filepath: str):
        """Save preprocessing state for later use."""
        state = {
            'normalization': self.normalization,
            'data_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in self.data_stats.items()},
            'class_weights': self.class_weights,
            'label_encoders': self.label_encoders,
            'fitted': self.fitted
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_preprocessing_state(self, filepath: str):
        """Load preprocessing state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.normalization = state['normalization']
        self.data_stats = {k: np.array(v) if isinstance(v, list) else v 
                          for k, v in state['data_stats'].items()}
        self.class_weights = state['class_weights']
        self.label_encoders = state['label_encoders']
        self.fitted = state['fitted']
