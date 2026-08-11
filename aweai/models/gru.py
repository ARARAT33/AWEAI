"""Gated Recurrent Unit (GRU) implemented from scratch in numpy."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid


class GRU(BaseModel):
    model_type = "gru"
    is_regressor = True

    def __init__(self, input_dim: int = 1, hidden: int = 16, output_dim: int = 1,
                 layers: int = 1, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, output_dim=output_dim,
                         layers=layers, **params)
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.output_dim = int(output_dim)
        self.layers = int(layers)
        self._build()

    def _build(self):
        rng = np.random.default_rng(7)
        d, h, o = self.input_dim, self.hidden, self.output_dim
        self.Wz = rng.normal(0, 0.1, (d, h)).astype(float)
        self.Uz = rng.normal(0, 0.1, (h, h)).astype(float)
        self.bz = np.zeros(h, dtype=float)
        self.Wr = rng.normal(0, 0.1, (d, h)).astype(float)
        self.Ur = rng.normal(0, 0.1, (h, h)).astype(float)
        self.br = np.zeros(h, dtype=float)
        self.Wh = rng.normal(0, 0.1, (d, h)).astype(float)
        self.Uh = rng.normal(0, 0.1, (h, h)).astype(float)
        self.bh = np.zeros(h, dtype=float)
        self.Why = rng.normal(0, 0.1, (h, o)).astype(float)
        self.by = np.zeros(o, dtype=float)

    def _step(self, x, h):
        z = sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
        r = sigmoid(x @ self.Wr + h @ self.Ur + self.br)
        h_hat = np.tanh(x @ self.Wh + (r * h) @ self.Uh + self.bh)
        h = (1 - z) * h + z * h_hat
        return h

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.01, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), -1, self.input_dim)
        y = np.asarray(y, dtype=float) if y is not None else X[:, -1, :self.output_dim].reshape(len(X), -1)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        B, T, D = X.shape
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(B):
                x = X[i]
                h = np.zeros(self.hidden)
                hs = [h.copy()]
                for t in range(T):
                    h = self._step(x[t], h)
                    hs.append(h.copy())
                y_pred = h @ self.Why + self.by
                target = y[i]
                loss = float(np.mean((y_pred - target) ** 2))
                epoch_loss += loss
                d_out = 2 * (y_pred - target) / max(target.size, 1)
                self.Why -= lr * np.outer(h, d_out.ravel())
                self.by -= lr * d_out.ravel()
                dh = (d_out @ self.Why.T).ravel()
                for t in reversed(range(T)):
                    x_t = x[t]
                    h_prev = hs[t]
                    h_cur = hs[t + 1]
                    z = sigmoid(x_t @ self.Wz + h_prev @ self.Uz + self.bz)
                    r = sigmoid(x_t @ self.Wr + h_prev @ self.Ur + self.br)
                    h_hat = np.tanh(x_t @ self.Wh + (r * h_prev) @ self.Uh + self.bh)
                    dz = (dh * (h_hat - h_prev) * z * (1 - z)).ravel()
                    dhh = (dh * z * (1 - h_hat ** 2)).ravel()
                    dr = ((dhh @ self.Uh.T) * h_prev * r * (1 - r)).ravel()
                    self.Wz -= lr * np.outer(x_t, dz)
                    self.Uz -= lr * np.outer(h_prev, dz)
                    self.bz -= lr * dz
                    self.Wr -= lr * np.outer(x_t, dr)
                    self.Ur -= lr * np.outer(h_prev, dr)
                    self.br -= lr * dr
                    self.Wh -= lr * np.outer(x_t, dhh)
                    self.Uh -= lr * np.outer(r * h_prev, dhh)
                    self.bh -= lr * dhh
                    dh = (dz @ self.Uz.T + dr @ self.Ur.T + (dhh @ self.Uh.T) * r).ravel()
            self.history["loss"].append(float(epoch_loss / max(B, 1)))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), -1, self.input_dim)
        out = []
        for i in range(len(X)):
            h = np.zeros(self.hidden)
            for t in range(X.shape[1]):
                h = self._step(X[i, t], h)
            out.append((h @ self.Why + self.by).ravel())
        return np.array(out)

    def forecast(self, X, steps: int = 1):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), -1, self.input_dim)
        seq = X[0].copy()
        out = []
        for _ in range(int(steps)):
            h = np.zeros(self.hidden)
            for t in range(len(seq)):
                h = self._step(seq[t], h)
            nxt = (h @ self.Why + self.by).ravel()
            out.append(nxt)
            seq = np.vstack([seq, nxt[:self.input_dim]])
        return np.array(out)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "Wz": self.Wz.tolist(), "Uz": self.Uz.tolist(), "bz": self.bz.tolist(),
            "Wr": self.Wr.tolist(), "Ur": self.Ur.tolist(), "br": self.br.tolist(),
            "Wh": self.Wh.tolist(), "Uh": self.Uh.tolist(), "bh": self.bh.tolist(),
            "Why": self.Why.tolist(), "by": self.by.tolist(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.Wz = np.asarray(state["Wz"], dtype=float)
        self.Uz = np.asarray(state["Uz"], dtype=float)
        self.bz = np.asarray(state["bz"], dtype=float)
        self.Wr = np.asarray(state["Wr"], dtype=float)
        self.Ur = np.asarray(state["Ur"], dtype=float)
        self.br = np.asarray(state["br"], dtype=float)
        self.Wh = np.asarray(state["Wh"], dtype=float)
        self.Uh = np.asarray(state["Uh"], dtype=float)
        self.bh = np.asarray(state["bh"], dtype=float)
        self.Why = np.asarray(state["Why"], dtype=float)
        self.by = np.asarray(state["by"], dtype=float)
        self.trained = True
