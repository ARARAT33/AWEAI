from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Task:
    id: str
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class MAML:
    def __init__(self, model: Any, inner_lr: float = 0.01, outer_lr: float = 0.001, inner_steps: int = 5) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self._meta_losses: List[float] = []

    def adapt(self, task: Task) -> Any:
        adapted = type(task.metadata.get("model_class", "Model"))()
        if hasattr(adapted, "weights"):
            adapted.weights = {k: v.copy() for k, v in self.model.weights.items()}
        for _ in range(self.inner_steps):
            if hasattr(adapted, "fit"):
                adapted.fit(task.support_x, task.support_y, epochs=1, lr=self.inner_lr)
        return adapted

    def meta_update(self, adapted_models: Sequence[Any], tasks: Sequence[Task]) -> float:
        total_meta_loss = 0.0
        for adapted, task in zip(adapted_models, tasks):
            if hasattr(adapted, "_forward"):
                logits, _ = adapted._forward(task.query_x)
                y = np.asarray(task.query_y, dtype=int)
                y_onehot = np.zeros((len(y), logits.shape[1]))
                y_onehot[np.arange(len(y)), y] = 1.0
                loss = -np.mean(np.sum(y_onehot * np.log_softmax(logits, axis=-1), axis=-1))
                total_meta_loss += float(loss)
        avg_meta_loss = total_meta_loss / max(len(adapted_models), 1)
        self._meta_losses.append(avg_meta_loss)
        if hasattr(self.model, "weights"):
            for key in self.model.weights:
                self.model.weights[key] -= self.outer_lr * self.model.weights[key]
        return avg_meta_loss

    def meta_train(self, tasks: Sequence[Task], epochs: int = 10) -> List[float]:
        for epoch in range(epochs):
            adapted_models = [self.adapt(task) for task in tasks]
            self.meta_update(adapted_models, tasks)
        return self._meta_losses

    def meta_losses(self) -> List[float]:
        return list(self._meta_losses)


class Reptile:
    def __init__(self, model: Any, inner_lr: float = 0.01, outer_lr: float = 0.001, inner_steps: int = 5) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self._losses: List[float] = []

    def adapt(self, task: Task) -> Any:
        adapted = type(task.metadata.get("model_class", "Model"))()
        if hasattr(adapted, "weights"):
            adapted.weights = {k: v.copy() for k, v in self.model.weights.items()}
        for _ in range(self.inner_steps):
            if hasattr(adapted, "fit"):
                adapted.fit(task.support_x, task.support_y, epochs=1, lr=self.inner_lr)
        return adapted

    def step(self, tasks: Sequence[Task]) -> float:
        total_loss = 0.0
        for task in tasks:
            adapted = self.adapt(task)
            if hasattr(adapted, "weights") and hasattr(self.model, "weights"):
                for key in self.model.weights:
                    delta = adapted.weights[key] - self.model.weights[key]
                    self.model.weights[key] += self.outer_lr * delta
            if hasattr(adapted, "_forward"):
                logits, _ = adapted._forward(task.query_x)
                y = np.asarray(task.query_y, dtype=int)
                y_onehot = np.zeros((len(y), logits.shape[1]))
                y_onehot[np.arange(len(y)), y] = 1.0
                loss = -np.mean(np.sum(y_onehot * np.log_softmax(logits, axis=-1), axis=-1))
                total_loss += float(loss)
        avg_loss = total_loss / max(len(tasks), 1)
        self._losses.append(avg_loss)
        return avg_loss

    def train(self, tasks: Sequence[Task], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            self.step(tasks)
        return self._losses

    def losses(self) -> List[float]:
        return list(self._losses)


class MetaLearner:
    def __init__(self, model: Any, meta_lr: float = 0.001, task_batch_size: int = 4) -> None:
        self.model = model
        self.meta_lr = meta_lr
        self.task_batch_size = task_batch_size
        self._history: List[float] = []

    def meta_step(self, tasks: Sequence[Task]) -> float:
        maml = MAML(model=self.model, inner_lr=0.01, outer_lr=self.meta_lr)
        adapted = [maml.adapt(task) for task in tasks[: self.task_batch_size]]
        loss = maml.meta_update(adapted, tasks[: self.task_batch_size])
        self._history.append(loss)
        return loss

    def train(self, task_generator: Callable[[], Sequence[Task]], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            tasks = task_generator()
            self.meta_step(tasks)
        return self._history

    def meta_history(self) -> List[float]:
        return list(self._history)
