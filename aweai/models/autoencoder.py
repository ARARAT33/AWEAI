"""Autoencoder (undercomplete) for anomaly detection / embedding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


class Autoencoder(BaseModel):
    model_type = "autoencoder"

    def __init__(self, input_dim: int = 2, hidden: Optional[List[int]] = None, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, **params)
        self.input_dim = int(input_dim)
        self.hidden = [int(h) for h in (hidden or [4, 2])]
        self.Ws: List[np.ndarray] = []
        self.bs: List[np.ndarray] = []
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(7)
        sizes = [self.input_dim] + self.hidden + [self.input_dim]
        self.Ws = [rng.normal(0, 0.1, (sizes[i], sizes[i + 1])).astype(float) for i in range(len(sizes) - 1)]
        self.bs = [np.zeros(s, dtype=float) for s in sizes[1:]]

    def encode(self, X):
        a = np.asarray(X, dtype=float)
        for i in range(len(self.Ws) - 1):
            a = np.maximum(a @ self.Ws[i] + self.bs[i], 0.0)
        return a

    def _forward(self, X):
        a = np.asarray(X, dtype=float)
        acts = [a]
        for i in range(len(self.Ws)):
            z = acts[-1] @ self.Ws[i] + self.bs[i]
            a = np.maximum(z, 0.0) if i < len(self.Ws) - 1 else z
            acts.append(a)
        return acts

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.01, **kw):
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        for _ in range(epochs):
            acts = self._forward(X)
            rec = acts[-1]
            loss = np.mean((rec - X) ** 2)
            d = 2 * (rec - X) / len(X)
            grads_w = [None] * len(self.Ws)
            grads_b = [None] * len(self.bs)
            for i in range(len(self.Ws) - 1, -1, -1):
                grads_w[i] = acts[i].T @ d
                grads_b[i] = d.sum(axis=0)
                if i > 0:
                    d = (d @ self.Ws[i].T) * (acts[i] > 0)
            for i in range(len(self.Ws)):
                self.Ws[i] -= lr * grads_w[i]
                self.bs[i] -= lr * grads_b[i]
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["reconstruction_loss"] = float(self.history["loss"][-1])
        return self

    def reconstruct(self, X):
        return self._forward(X)[-1]

    def anomaly_score(self, X):
        rec = self.reconstruct(X)
        return np.mean((np.asarray(X, dtype=float) - rec) ** 2, axis=1)

    def predict(self, X):
        return self.reconstruct(X)

    def state_dict(self):
        return {"Ws": [w.tolist() for w in self.Ws], "bs": [b.tolist() for b in self.bs]}

    def load_state(self, state):
        self.Ws = [np.asarray(w, dtype=float) for w in state["Ws"]]
        self.bs = [np.asarray(b, dtype=float) for b in state["bs"]]
        self.trained = True
