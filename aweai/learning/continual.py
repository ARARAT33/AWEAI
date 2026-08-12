from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class EWC:
    def __init__(self, model: Any, importance: float = 1000.0) -> None:
        self.model = model
        self.importance = importance
        self._fisher: Dict[str, np.ndarray] = {}
        self._optimal_params: Dict[str, np.ndarray] = {}

    def compute_fisher(self, data: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        if not hasattr(self.model, "weights"):
            return
        self._fisher = {k: np.zeros_like(v) for k, v in self.model.weights.items()}
        for x, y in data:
            if hasattr(self.model, "_forward"):
                logits, cache = self.model._forward(x)
                probs = np.softmax(logits, axis=-1)
                loss = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=-1))
                for key in self._fisher:
                    self._fisher[key] += np.ones_like(self._fisher[key])
        for key in self._fisher:
            self._fisher[key] /= max(len(data), 1)

    def register_task(self, data: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        if hasattr(self.model, "weights"):
            self._optimal_params = {k: v.copy() for k, v in self.model.weights.items()}
        self.compute_fisher(data)

    def penalty(self) -> float:
        if not self._fisher or not self._optimal_params:
            return 0.0
        penalty = 0.0
        for key in self._optimal_params:
            if key in self._fisher and key in self._optimal_params:
                penalty += float(np.sum(self._fisher[key] * (self.model.weights[key] - self._optimal_params[key]) ** 2))
        return self.importance * penalty

    def loss_with_penalty(self, base_loss: float) -> float:
        return base_loss + self.penalty()


class ReplayBuffer:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._buffer: List[Dict[str, Any]] = []
        self._position = 0

    def add(self, experience: Dict[str, Any]) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
        else:
            self._buffer[self._position] = experience
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class ContinualLearner:
    def __init__(self, model: Any, strategy: str = "ewc") -> None:
        self.model = model
        self.strategy = strategy
        self._ewc = EWC(model=model) if strategy == "ewc" else None
        self._replay = ReplayBuffer() if strategy == "replay" else None
        self._tasks_seen: List[str] = []
        self._performance: Dict[str, List[float]] = {}

    def learn_task(self, task_id: str, data: Sequence[Tuple[np.ndarray, np.ndarray]], epochs: int = 5) -> Dict[str, Any]:
        self._tasks_seen.append(task_id)
        if self.strategy == "ewc" and self._ewc:
            self._ewc.register_task(data)
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in data:
                if hasattr(self.model, "fit"):
                    result = self.model.fit(x, y, epochs=1)
                    total_loss += result.get("loss", 0.0)
                if self._replay:
                    self._replay.add({"x": x, "y": y, "task": task_id})
            avg_loss = total_loss / max(len(data), 1)
            history.append({"epoch": epoch, "loss": avg_loss})
        self._performance[task_id] = [h["loss"] for h in history]
        return {"task_id": task_id, "epochs": epochs, "history": history, "final_loss": history[-1]["loss"] if history else 0.0}

    def evaluate_task(self, task_id: str, data: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
        correct = 0
        total = 0
        for x, y in data:
            if hasattr(self.model, "predict"):
                preds = self.model.predict(x)
                correct += int(np.mean(np.asarray(preds) == np.asarray(y)))
                total += 1
        return correct / max(total, 1)

    def catastrophic_forgetting(self, task_id: str) -> float:
        if task_id not in self._performance:
            return 0.0
        initial = self._performance[task_id][0] if self._performance[task_id] else 1.0
        current = self._performance[task_id][-1] if self._performance[task_id] else 1.0
        return max(0.0, (initial - current) / max(initial, 1e-8))

    def tasks_seen(self) -> List[str]:
        return list(self._tasks_seen)

    def performance_history(self) -> Dict[str, List[float]]:
        return dict(self._performance)
