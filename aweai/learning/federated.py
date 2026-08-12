from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClientUpdate:
    client_id: str
    weights: Dict[str, np.ndarray]
    num_samples: int
    loss: float


class FederatedAveraging:
    def __init__(self, global_model: Any, num_clients: int = 10, client_fraction: float = 0.5, lr: float = 1.0) -> None:
        self.global_model = global_model
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.lr = lr
        self._rng = np.random.default_rng(42)
        self._round_history: List[Dict[str, Any]] = []

    def select_clients(self) -> List[str]:
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        return [f"client_{i}" for i in range(num_selected)]

    def aggregate(self, updates: Sequence[ClientUpdate]) -> None:
        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            return
        if not hasattr(self.global_model, "weights"):
            return
        for key in self.global_model.weights:
            weighted_sum = sum(u.weights[key] * u.num_samples for u in updates)
            self.global_model.weights[key] = self.lr * weighted_sum / total_samples

    def train_round(self, client_data: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]], train_fn: Callable[[Any, Sequence[Tuple[np.ndarray, np.ndarray]], int], ClientUpdate]) -> Dict[str, Any]:
        selected = self.select_clients()
        updates = []
        for client_id in selected:
            if client_id in client_data:
                update = train_fn(self.global_model, client_data[client_id], 1)
                updates.append(update)
        self.aggregate(updates)
        round_info = {"round": len(self._round_history), "clients": len(selected), "updates": len(updates), "avg_loss": float(np.mean([u.loss for u in updates])) if updates else 0.0}
        self._round_history.append(round_info)
        return round_info

    def train(self, client_data: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]], train_fn: Callable[[Any, Sequence[Tuple[np.ndarray, np.ndarray]], int], ClientUpdate], rounds: int = 10) -> List[Dict[str, Any]]:
        for _ in range(rounds):
            self.train_round(client_data, train_fn)
        return self._round_history

    def evaluate(self, test_data: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
        if not hasattr(self.global_model, "predict"):
            return 0.0
        correct = 0
        for x, y in test_data:
            preds = self.global_model.predict(x)
            correct += int(np.mean(np.asarray(preds) == np.asarray(y)))
        return correct / max(len(test_data), 1)

    def round_history(self) -> List[Dict[str, Any]]:
        return list(self._round_history)


class FedProx:
    def __init__(self, global_model: Any, mu: float = 0.01, **kwargs: Any) -> None:
        self.global_model = global_model
        self.mu = mu
        self.fed_avg = FederatedAveraging(global_model=global_model, **kwargs)

    def proximal_term(self, local_weights: Dict[str, np.ndarray]) -> float:
        if not hasattr(self.global_model, "weights"):
            return 0.0
        penalty = 0.0
        for key in self.global_model.weights:
            penalty += float(np.sum((local_weights[key] - self.global_model.weights[key]) ** 2))
        return 0.5 * self.mu * penalty

    def train_round(self, client_data: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]], train_fn: Callable[[Any, Sequence[Tuple[np.ndarray, np.ndarray]], int], ClientUpdate]) -> Dict[str, Any]:
        return self.fed_avg.train_round(client_data, train_fn)

    def train(self, client_data: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]], train_fn: Callable[[Any, Sequence[Tuple[np.ndarray, np.ndarray]], int], ClientUpdate], rounds: int = 10) -> List[Dict[str, Any]]:
        return self.fed_avg.train(client_data, train_fn, rounds=rounds)


class SCAFFOLD:
    def __init__(self, global_model: Any, lr: float = 1.0) -> None:
        self.global_model = global_model
        self.lr = lr
        self._server_control: Dict[str, np.ndarray] = {}
        self._client_controls: Dict[str, Dict[str, np.ndarray]] = {}
        self.fed_avg = FederatedAveraging(global_model=global_model)

    def server_control(self) -> Dict[str, np.ndarray]:
        if not hasattr(self.global_model, "weights"):
            return {}
        if not self._server_control:
            self._server_control = {k: np.zeros_like(v) for k, v in self.global_model.weights.items()}
        return self._server_control

    def client_control(self, client_id: str) -> Dict[str, np.ndarray]:
        if client_id not in self._client_controls:
            if hasattr(self.global_model, "weights"):
                self._client_controls[client_id] = {k: np.zeros_like(v) for k, v in self.global_model.weights.items()}
            else:
                self._client_controls[client_id] = {}
        return self._client_controls[client_id]

    def train_round(self, client_data: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]], train_fn: Callable[[Any, Sequence[Tuple[np.ndarray, np.ndarray]], int], ClientUpdate]) -> Dict[str, Any]:
        selected = self.fed_avg.select_clients()
        updates = []
        for client_id in selected:
            if client_id in client_data:
                update = train_fn(self.global_model, client_data[client_id], 1)
                sc = self.server_control()
                cc = self.client_control(client_id)
                for key in sc:
                    update.weights[key] = self.global_model.weights[key] - self.lr * (update.weights[key] - self.global_model.weights[key] + sc[key] - cc.get(key, sc[key]))
                updates.append(update)
                for key in sc:
                    cc[key] = cc.get(key, np.zeros_like(sc[key])) - (1 / self.lr) * (self.global_model.weights[key] - update.weights[key]) + sc[key]
        if updates:
            for key in self._server_control:
                self._server_control[key] = sum(u.weights[key] for u in updates) / len(updates)
        self.fed_avg.aggregate(updates)
        return {"round": len(self.fed_avg._round_history), "clients": len(selected)}
