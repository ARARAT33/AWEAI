#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Module - Մոդելների ուսուցում և կառավարում
Advanced model training, fine-tuning and evaluation
"""

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import hashlib


class ModelTrainer:
    """Հզոր մոդելների ուսուցման գործիք"""
    
    def __init__(self, model_type: str = "sklearn", 
                 storage_path: str = "./output/models"):
        self.model_type = model_type
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.training_history = []
        self.metrics = {}
        
    def create_model(self, model_name: str, **kwargs) -> Any:
        """Ստեղծել մոդել ըստ տեսակի"""
        if self.model_type == "sklearn":
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            
            models = {
                'logistic_regression': LogisticRegression(**kwargs),
                'linear_regression': LinearRegression(**kwargs),
                'random_forest_classifier': RandomForestClassifier(**kwargs),
                'random_forest_regressor': RandomForestRegressor(**kwargs),
                'svc': SVC(**kwargs),
                'svr': SVR(**kwargs),
                'mlp_classifier': MLPClassifier(**kwargs),
                'mlp_regressor': MLPRegressor(**kwargs)
            }
            
            self.model = models.get(model_name)
            if not self.model:
                raise ValueError(f"Unknown model: {model_name}")
                
        elif self.model_type == "tensorflow":
            import tensorflow as tf
            # TensorFlow/Keras model creation logic here
            pass
            
        elif self.model_type == "pytorch":
            import torch
            import torch.nn as nn
            # PyTorch model creation logic here
            pass
        
        return self.model
    
    def train(self, X_train: Any, y_train: Any, 
              X_val: Optional[Any] = None, 
              y_val: Optional[Any] = None,
              epochs: int = 10, batch_size: int = 32,
              callbacks: Optional[List[Callable]] = None) -> Dict:
        """Ուսուցանել մոդելը"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")
        
        start_time = datetime.now()
        
        try:
            if self.model_type == "sklearn":
                self.model.fit(X_train, y_train)
                train_score = self.model.score(X_train, y_train)
                
                val_score = None
                if X_val is not None and y_val is not None:
                    val_score = self.model.score(X_val, y_val)
                
                self.metrics = {
                    'train_score': train_score,
                    'validation_score': val_score,
                    'training_samples': len(X_train),
                    'training_time': (datetime.now() - start_time).total_seconds()
                }
                
            elif self.model_type == "tensorflow":
                # TensorFlow training logic
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks
                )
                self.metrics = {
                    'history': history.history,
                    'epochs': epochs
                }
                
            elif self.model_type == "pytorch":
                # PyTorch training logic
                pass
            
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type,
                'metrics': self.metrics,
                'duration': (datetime.now() - start_time).total_seconds()
            }
            self.training_history.append(training_record)
            
            return self.metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate(self, X_test: Any, y_test: Any) -> Dict:
        """Գնահատել մոդելի որակը"""
        if self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            if self.model_type == "sklearn":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                y_pred = self.model.predict(X_test)
                
                # Classification metrics
                try:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    eval_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                except:
                    # Regression metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    eval_metrics = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    }
                
                return eval_metrics
                
        except Exception as e:
            return {'error': str(e)}
    
    def save_model(self, filename: str) -> str:
        """Պահպանել մոդելը"""
        if self.model is None:
            return "No model to save"
        
        model_path = self.storage_path / filename
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        return str(model_path)
    
    def load_model(self, filename: str) -> Any:
        """Բեռնել պահպանված մոդել"""
        model_path = self.storage_path / filename
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_type = data['model_type']
            self.metrics = data.get('metrics', {})
        
        return self.model
    
    def predict(self, X: Any) -> Any:
        """Կատարել կանխատեսում"""
        if self.model is None:
            return None
        return self.model.predict(X)
    
    def get_training_history(self) -> List[Dict]:
        """Ստանալ ուսուցման պատմությունը"""
        return self.training_history


class HyperparameterTuner:
    """Հիպերպարամետրերի օպտիմալացում"""
    
    def __init__(self, model_trainer: ModelTrainer):
        self.trainer = model_trainer
        self.best_params = {}
        self.best_score = 0
        
    def grid_search(self, X_train: Any, y_train: Any, 
                   param_grid: Dict[str, List], cv_folds: int = 5) -> Dict:
        """Grid Search հիպերպարամետրերի ընտրության համար"""
        from itertools import product
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        best_params = {}
        best_score = 0
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            try:
                self.trainer.create_model(self.trainer.model.__class__.__name__, **params)
                scores = cross_val_score(
                    self.trainer.model, X_train, y_train, 
                    cv=cv_folds, scoring='accuracy'
                )
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
            except Exception as e:
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': float(best_score),
            'total_combinations': len(list(product(*param_values)))
        }
    
    def random_search(self, X_train: Any, y_train: Any,
                     param_distributions: Dict[str, List],
                     n_iterations: int = 10) -> Dict:
        """Random Search հիպերպարամետրերի ընտրության համար"""
        import random
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        best_params = {}
        best_score = 0
        
        for _ in range(n_iterations):
            params = {
                key: random.choice(values)
                for key, values in param_distributions.items()
            }
            
            try:
                self.trainer.create_model(self.trainer.model.__class__.__name__, **params)
                scores = cross_val_score(
                    self.trainer.model, X_train, y_train,
                    cv=5, scoring='accuracy'
                )
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
            except Exception as e:
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': float(best_score),
            'iterations': n_iterations
        }


class ModelEvaluator:
    """Մոդելների համապարփակ գնահատում"""
    
    @staticmethod
    def classification_report(y_true: List, y_pred: List) -> Dict:
        """Մանրամասն classification report"""
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'report': report,
            'confusion_matrix': cm.tolist()
        }
    
    @staticmethod
    def regression_report(y_true: List, y_pred: List) -> Dict:
        """Մանրամասն regression report"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'explained_variance': float(np.var(y_true - y_pred) / np.var(y_true))
        }
    
    @staticmethod
    def cross_validation_score(model: Any, X: Any, y: Any, 
                              cv_folds: int = 5, scoring: str = 'accuracy') -> Dict:
        """Cross-validation գնահատում"""
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        
        return {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores.tolist(),
            'cv_folds': cv_folds
        }


__all__ = ['ModelTrainer', 'HyperparameterTuner', 'ModelEvaluator']
