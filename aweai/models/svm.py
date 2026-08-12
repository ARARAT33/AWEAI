# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Linear SVM from scratch via primal SGD (hinge loss).

A margin-maximizing linear classifier trained with stochastic gradient
descent — very small memory (a single weight vector + bias) and fast on CPU.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid


class SVM(BaseModel):
    """Linear Support Vector Machine trained with SGD on the hinge loss."""

    model_type = "svm"
    is_classifier = True

    def __init__(self, input_dim: int = 2, C: float = 1.0, lr: float = 0.01,
                 epochs: int = 100, seed: int = 0, **params):
        super().__init__(input_dim=input_dim, C=float(C), lr=float(lr),
                         epochs=int(epochs), seed=seed, **params)
        self.C = float(C)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self._seed = int(seed)
        self.W: np.ndarray = np.array([])
        self.b: float = 0.0
        self.classes_: np.ndarray = np.array([])

    def fit(self, X, y=None, epochs: int = None, lr: float = None, C: float = None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        # binary labels -> {-1, +1}
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            y_bin = np.where(y == self.classes_[0], -1.0, 1.0)
        else:
            y_bin = y.astype(float)
        epochs = int(kw.get("epochs", epochs or self.epochs))
        lr = float(kw.get("lr", lr or self.lr))
        C = float(kw.get("C", C or self.C))
        rng = np.random.default_rng(self._seed)
        self.W = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(epochs):
            perm = rng.permutation(len(X))
            for i in perm:
                xi = X[i]
                yi = y_bin[i]
                margin = yi * (xi @ self.W + self.b)
                if margin < 1.0:
                    self.W -= lr * (-C * yi * xi + (1.0 / len(X)) * self.W)
                    self.b -= lr * (-C * yi)
                else:
                    self.W -= lr * (1.0 / len(X)) * self.W
        self.trained = True
        self.metrics["accuracy"] = float(np.mean(self.predict(X) == y))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scores = X @ self.W + self.b
        if len(self.classes_) == 2:
            pred = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            pred = np.where(scores >= 0, 1, -1)
        return np.asarray(pred)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scores = X @ self.W + self.b
        p_pos = sigmoid(scores).reshape(-1, 1)
        p_neg = 1.0 - p_pos
        return np.hstack([p_neg, p_pos])

    def state_dict(self) -> Dict[str, Any]:
        return {"W": self.W.tolist(), "b": self.b, "classes": self.classes_.tolist()}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.W = np.asarray(state["W"], dtype=float)
        self.b = float(state["b"])
        self.classes_ = np.asarray(state.get("classes", []), dtype=int)
        self.trained = True
