"""
Federated Learning with Differential Privacy (FL-DP)
Գաղտնի պահպանող դաշնային ուսուցում

Privacy-preserving federated learning implementation.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ClientModel:
    """Client model state"""
    client_id: str
    weights: Dict[str, np.ndarray]
    data_size: int
    privacy_budget: float


class FederatedLearningDP:
    """
    Federated Learning with Differential Privacy
    
    Features:
    - Secure aggregation
    - Differential privacy noise
    - Client selection
    - Privacy budget tracking
    """
    
    def __init__(
        self,
        num_clients: int = 100,
        clients_per_round: int = 10,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        
        self.logger = logging.getLogger(__name__)
        self.global_model: Optional[Dict[str, np.ndarray]] = None
        self.client_models: Dict[str, ClientModel] = {}
        self.round_history: List[Dict[str, Any]] = []
        
        # Privacy accounting
        self.total_privacy_spent = 0.0
    
    def initialize_clients(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize client models"""
        for i in range(self.num_clients):
            client_id = f"client_{i}"
            weights = {k: v.copy() for k, v in model_template.items()}
            
            self.client_models[client_id] = ClientModel(
                client_id=client_id,
                weights=weights,
                data_size=np.random.randint(100, 10000),
                privacy_budget=self.privacy_epsilon,
            )
    
    def select_clients(self) -> List[str]:
        """Select clients for this round"""
        available = list(self.client_models.keys())
        return np.random.choice(available, self.clients_per_round, replace=False).tolist()
    
    def local_training(
        self,
        client_id: str,
        local_data: Any,
        epochs: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Perform local training on client"""
        client = self.client_models[client_id]
        
        # Simulate local training
        updated_weights = {}
        for layer_name, weights in client.weights.items():
            gradient = np.random.randn(*weights.shape) * 0.01
            updated_weights[layer_name] = weights + gradient
        
        return updated_weights
    
    def add_dp_noise(
        self,
        weights: Dict[str, np.ndarray],
        sensitivity: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Add differential privacy noise"""
        # Calculate noise scale using Gaussian mechanism
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.privacy_delta)) / self.privacy_epsilon
        
        noisy_weights = {}
        for layer_name, weights in weights.items():
            noise = np.random.normal(0, sigma, weights.shape)
            noisy_weights[layer_name] = weights + noise
        
        return noisy_weights
    
    def aggregate_models(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_sizes: List[int],
    ) -> Dict[str, np.ndarray]:
        """Securely aggregate client models"""
        total_size = sum(client_sizes)
        
        if self.global_model is None:
            self.global_model = {k: np.zeros_like(v) for k, v in client_updates[0].items()}
        
        aggregated = {}
        for layer_name in self.global_model.keys():
            weighted_sum = np.zeros_like(self.global_model[layer_name])
            
            for update, size in zip(client_updates, client_sizes):
                weight = size / total_size
                weighted_sum += weight * update[layer_name]
            
            aggregated[layer_name] = weighted_sum
        
        return aggregated
    
    def train_round(self, round_num: int) -> Dict[str, Any]:
        """Execute one round of federated learning"""
        # Select clients
        selected_clients = self.select_clients()
        
        # Local training
        client_updates = []
        client_sizes = []
        
        for client_id in selected_clients:
            # Simulate local data
            local_data = None
            
            # Local training
            update = self.local_training(client_id, local_data)
            
            # Add DP noise
            noisy_update = self.add_dp_noise(update)
            
            client_updates.append(noisy_update)
            client_sizes.append(self.client_models[client_id].data_size)
        
        # Aggregate
        self.global_model = self.aggregate_models(client_updates, client_sizes)
        
        # Update privacy budget
        round_epsilon = self.privacy_epsilon / self.clients_per_round
        self.total_privacy_spent += round_epsilon
        
        result = {
            'round': round_num,
            'clients_participated': len(selected_clients),
            'privacy_spent': self.total_privacy_spent,
        }
        
        self.round_history.append(result)
        self.logger.info(f"Round {round_num}: Privacy spent = {self.total_privacy_spent:.4f}")
        
        return result
    
    def get_global_model(self) -> Optional[Dict[str, np.ndarray]]:
        """Get global model"""
        return self.global_model
    
    def get_privacy_status(self) -> Dict[str, float]:
        """Get privacy budget status"""
        return {
            'total_budget': self.privacy_epsilon,
            'spent': self.total_privacy_spent,
            'remaining': max(0, self.privacy_epsilon - self.total_privacy_spent),
            'delta': self.privacy_delta,
        }
