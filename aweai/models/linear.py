"""Linear models: LinearRegression (closed form) and LogisticRegression (SGD)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid


class LinearRegression(BaseModel):
    model_type = "linear"
    is_regressor = True

    def __init__(self, **params):
        super().__init__(**params)
        self.coef_: np.ndarray = np.zeros(1)
        self.intercept_: float = 0.0

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        coefs, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.intercept_ = float(coefs[0])
        self.coef_ = coefs[1:]
        self.trained = True
        pred = self.predict(X)
        self.metrics["mse"] = float(np.mean((pred - y) ** 2))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def state_dict(self):
        return {"coef": self.coef_.tolist(), "intercept": self.intercept_}

    def load_state(self, state):
        self.coef_ = np.asarray(state["coef"], dtype=float)
        self.intercept_ = float(state["intercept"])
        self.trained = True


class LogisticRegression(BaseModel):
    model_type = "logistic"
    is_classifier = True

    def __init__(self, input_dim: int = 2, **params):
        super().__init__(input_dim=input_dim, **params)
        self.W: np.ndarray = np.zeros(input_dim)
        self.b: float = 0.0

    def fit(self, X, y=None, epochs: int = 50, lr: float = 0.05, **kw):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        self.W = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(epochs):
            z = X @ self.W + self.b
            p = sigmoid(z)
            grad = (p - y) / len(y)
            self.W -= lr * (X.T @ grad)
            self.b -= lr * float(grad.sum())
        self.trained = True
        self.metrics["final_loss"] = float(np.mean(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (sigmoid(X @ self.W + self.b) >= 0.5).astype(int)

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.W + self.b)

    def state_dict(self):
        return {"W": self.W.tolist(), "b": self.b}

    def load_state(self, state):
        self.W = np.asarray(state["W"], dtype=float)
        self.b = float(state["b"])
        self.trained = True
