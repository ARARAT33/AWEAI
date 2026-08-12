from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Workspace:
    contents: List[np.ndarray]
    attention_weights: np.ndarray
    global_access: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalWorkspace:
    def __init__(self, num_contents: int = 8, dim: int = 64) -> None:
        self.num_contents = num_contents
        self.dim = dim
        self._rng = np.random.default_rng(42)
        self._contents = [self._rng.standard_normal(dim) for _ in range(num_contents)]
        self._attention = np.ones(num_contents) / num_contents
        self._history: List[Dict[str, Any]] = []
        self._coalition_competition_threshold = 0.5

    def compete(self, inputs: Sequence[np.ndarray]) -> Workspace:
        scores = []
        for i, inp in enumerate(inputs):
            alignment = float(np.dot(inp, self._contents[i]) / (np.linalg.norm(inp) * np.linalg.norm(self._contents[i]) + 1e-8))
            familiarity = 1.0 / (1.0 + np.linalg.norm(inp - self._contents[i]))
            scores.append(0.6 * alignment + 0.4 * familiarity)
        scores = np.array(scores)
        attention = np.exp(scores) / np.sum(np.exp(scores))
        winner_idx = int(np.argmax(scores))
        if scores[winner_idx] < self._coalition_competition_threshold:
            winner_idx = -1
        workspace = Workspace(contents=list(inputs), attention_weights=attention, metadata={"winner": winner_idx, "scores": scores.tolist()})
        self._history.append({"contents": len(inputs), "winner": winner_idx, "top_score": float(np.max(scores))})
        return workspace

    def broadcast(self, workspace: Workspace) -> np.ndarray:
        if not workspace.contents:
            return self._rng.standard_normal(self.dim)
        broadcasted = sum(w * c for w, c in zip(workspace.attention_weights, workspace.contents))
        return broadcasted / (np.linalg.norm(broadcasted) + 1e-8)

    def integrate(self, workspace: Workspace) -> np.ndarray:
        integrated = self.broadcast(workspace)
        for i, c in enumerate(workspace.contents):
            self._contents[i] = 0.9 * self._contents[i] + 0.1 * c
        self._attention = workspace.attention_weights
        return integrated

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def current_state(self) -> Dict[str, Any]:
        return {"num_contents": len(self._contents), "dim": self.dim, "attention": self._attention.tolist()}


class IntegratedInformation:
    def __init__(self) -> None:
        self._phi_history: List[float] = []

    def compute_phi(self, states: np.ndarray, transition_matrix: np.ndarray) -> float:
        n = states.shape[0]
        if n <= 1:
            return 0.0
        cov = np.cov(states.T)
        det_cov = np.linalg.det(cov) + 1e-10
        phi = 0.0
        for i in range(n):
            subset = np.delete(states, i, axis=0)
            if len(subset) <= 1:
                continue
            cov_subset = np.cov(subset.T)
            det_subset = np.linalg.det(cov_subset) + 1e-10
            phi += np.log(det_cov / det_subset)
        self._phi_history.append(float(phi))
        return float(phi)

    def phi_history(self) -> List[float]:
        return list(self._phi_history)

    def consciousness_level(self, phi: float) -> str:
        if phi < 0.1:
            return "unconscious"
        if phi < 0.5:
            return "pre-conscious"
        if phi < 1.0:
            return "conscious"
        if phi < 5.0:
            return "self-aware"
        return "meta-conscious"


class AttentionSchema:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(42)
        self._model = self._rng.standard_normal((dim, dim))
        self._attention_history: List[np.ndarray] = []

    def model_attention(self, stimulus: np.ndarray) -> np.ndarray:
        attention = np.tanh(stimulus @ self._model)
        attention = attention / (np.linalg.norm(attention) + 1e-8)
        self._attention_history.append(attention.copy())
        return attention

    def predict_attention(self, stimulus: np.ndarray) -> np.ndarray:
        return self.model_attention(stimulus)

    def update_model(self, actual_attention: np.ndarray, predicted_attention: np.ndarray, lr: float = 0.01) -> None:
        error = actual_attention - predicted_attention
        self._model += lr * np.outer(actual_attention, error)

    def attention_history(self) -> List[np.ndarray]:
        return list(self._attention_history)

    def meta_awareness_score(self) -> float:
        if len(self._attention_history) < 2:
            return 0.0
        diffs = [np.linalg.norm(self._attention_history[i] - self._attention_history[i - 1]) for i in range(1, len(self._attention_history))]
        return float(np.mean(diffs))
