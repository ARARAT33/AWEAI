from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class SimCLR:
    def __init__(self, model: Any, temperature: float = 0.07, proj_dim: int = 128) -> None:
        self.model = model
        self.temperature = temperature
        self.proj_dim = proj_dim
        self._rng = np.random.default_rng(42)
        self._projection = self._rng.standard_normal((64, proj_dim)) * 0.1
        self._losses: List[float] = []

    def _project(self, h: np.ndarray) -> np.ndarray:
        return np.tanh(h @ self._projection)

    def contrastive_loss(self, z1: np.ndarray, z2: np.ndarray) -> float:
        z1 = z1 / (np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
        z2 = z2 / (np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)
        logits = np.vstack([z1 @ z2.T, z1 @ z2.T]) / self.temperature
        labels = np.arange(len(z1))
        loss = 0.0
        for i in range(len(z1)):
            exp_logits = np.exp(logits[i])
            loss += -np.log(exp_logits[i] / np.sum(exp_logits)) - np.log(exp_logits[len(z1) + i] / np.sum(exp_logits))
        return float(loss / len(z1))

    def step(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if hasattr(self.model, "_forward"):
            h1, _ = self.model._forward(x1)
            h2, _ = self.model._forward(x2)
            z1 = self._project(h1)
            z2 = self._project(h2)
            loss = self.contrastive_loss(z1, z2)
            self._losses.append(loss)
            return loss
        return 0.0

    def train(self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            for x1, x2 in pairs:
                self.step(x1, x2)
        return self._losses

    def losses(self) -> List[float]:
        return list(self._losses)


class MoCo:
    def __init__(self, model: Any, queue_size: int = 65536, momentum: float = 0.999, temperature: float = 0.07) -> None:
        self.model = model
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self._rng = np.random.default_rng(42)
        self._queue: List[np.ndarray] = []
        self._momentum_model = type("MomentumModel", (), {"weights": {}})()
        if hasattr(model, "weights"):
            self._momentum_model.weights = {k: v.copy() for k, v in model.weights.items()}
        self._losses: List[float] = []

    def _enqueue(self, z: np.ndarray) -> None:
        if len(self._queue) >= self.queue_size:
            self._queue.pop(0)
        self._queue.append(z.copy())

    def contrastive_loss(self, q: np.ndarray, k: np.ndarray, queue: Sequence[np.ndarray]) -> float:
        q = q / (np.linalg.norm(q) + 1e-8)
        k = k / (np.linalg.norm(k) + 1e-8)
        logits = np.array([np.dot(q, k)] + [np.dot(q, z) for z in queue]) / self.temperature
        return float(-np.log(np.exp(logits[0]) / np.sum(np.exp(logits))))

    def step(self, x_q: np.ndarray, x_k: np.ndarray) -> float:
        if hasattr(self.model, "_forward"):
            q, _ = self.model._forward(x_q)
            with np.errstate(all="ignore"):
                k, _ = self.model._forward(x_k)
            loss = self.contrastive_loss(q, k, self._queue)
            self._enqueue(k)
            self._losses.append(loss)
            return loss
        return 0.0

    def update_momentum(self) -> None:
        if hasattr(self.model, "weights"):
            for key in self.model.weights:
                self._momentum_model.weights[key] = self.momentum * self._momentum_model.weights[key] + (1 - self.momentum) * self.model.weights[key]

    def train(self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            for x_q, x_k in pairs:
                self.step(x_q, x_k)
                self.update_momentum()
        return self._losses

    def losses(self) -> List[float]:
        return list(self._losses)


class BYOL:
    def __init__(self, model: Any, target_model: Any, momentum: float = 0.996) -> None:
        self.model = model
        self.target_model = target_model
        self.momentum = momentum
        self._losses: List[float] = []

    def step(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if hasattr(self.model, "_forward") and hasattr(self.target_model, "_forward"):
            z1_online, _ = self.model._forward(x1)
            z2_target, _ = self.target_model._forward(x2)
            z1_online = z1_online / (np.linalg.norm(z1_online) + 1e-8)
            z2_target = z2_target / (np.linalg.norm(z2_target) + 1e-8)
            loss = float(np.mean((z1_online - z2_target) ** 2))
            self._losses.append(loss)
            return loss
        return 0.0

    def update_target(self) -> None:
        if hasattr(self.target_model, "weights") and hasattr(self.model, "weights"):
            for key in self.target_model.weights:
                self.target_model.weights[key] = self.momentum * self.target_model.weights[key] + (1 - self.momentum) * self.model.weights[key]

    def train(self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            for x1, x2 in pairs:
                self.step(x1, x2)
                self.update_target()
        return self._losses

    def losses(self) -> List[float]:
        return list(self._losses)


class DINO:
    def __init__(self, student: Any, teacher: Any, momentum: float = 0.996, temperature_teacher: float = 0.04, temperature_student: float = 0.1) -> None:
        self.student = student
        self.teacher = teacher
        self.momentum = momentum
        self.temperature_teacher = temperature_teacher
        self.temperature_student = temperature_student
        self._losses: List[float] = []

    def step(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if hasattr(self.student, "_forward") and hasattr(self.teacher, "_forward"):
            s1, _ = self.student._forward(x1)
            s2, _ = self.student._forward(x2)
            t1, _ = self.teacher._forward(x1)
            t2, _ = self.teacher._forward(x2)
            loss = 0.5 * self._softcrossentropy(s1 / self.temperature_student, np.softmax(t1 / self.temperature_teacher))
            loss += 0.5 * self._softcrossentropy(s2 / self.temperature_student, np.softmax(t2 / self.temperature_teacher))
            self._losses.append(float(loss))
            return float(loss)
        return 0.0

    def _softcrossentropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        log_probs = np.log_softmax(logits, axis=-1)
        return float(-np.sum(targets * log_probs) / len(logits))

    def update_teacher(self) -> None:
        if hasattr(self.teacher, "weights") and hasattr(self.student, "weights"):
            for key in self.teacher.weights:
                self.teacher.weights[key] = self.momentum * self.teacher.weights[key] + (1 - self.momentum) * self.student.weights[key]

    def train(self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            for x1, x2 in pairs:
                self.step(x1, x2)
                self.update_teacher()
        return self._losses

    def losses(self) -> List[float]:
        return list(self._losses)


class MAE:
    def __init__(self, model: Any, mask_ratio: float = 0.75, decoder_dim: int = 64) -> None:
        self.model = model
        self.mask_ratio = mask_ratio
        self.decoder_dim = decoder_dim
        self._rng = np.random.default_rng(42)
        self._losses: List[float] = []

    def mask(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, D = x.shape
        num_mask = int(D * self.mask_ratio)
        mask = np.zeros(D, dtype=bool)
        mask[self._rng.choice(D, num_mask, replace=False)] = True
        masked_x = x.copy()
        masked_x[:, mask] = 0
        return masked_x, mask, x

    def step(self, x: np.ndarray) -> float:
        masked_x, mask, target = self.mask(x)
        if hasattr(self.model, "_forward"):
            pred, _ = self.model._forward(masked_x)
            loss = float(np.mean((pred - target) ** 2))
            self._losses.append(loss)
            return loss
        return 0.0

    def train(self, data: Sequence[np.ndarray], epochs: int = 10) -> List[float]:
        for _ in range(epochs):
            for x in data:
                self.step(x)
        return self._losses

    def losses(self) -> List[float]:
        return list(self._losses)
