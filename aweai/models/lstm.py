"""Long Short-Term Memory (LSTM) implemented from scratch in numpy."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid, softmax


class LSTM(BaseModel):
    model_type = "lstm"

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
        rng = np.random.default_rng(42)
        d, h, o = self.input_dim, self.hidden, self.output_dim
        self.Wf = rng.normal(0, 0.1, (d + h, h)).astype(float)
        self.Wi = rng.normal(0, 0.1, (d + h, h)).astype(float)
        self.Wc = rng.normal(0, 0.1, (d + h, h)).astype(float)
        self.Wo = rng.normal(0, 0.1, (d + h, h)).astype(float)
        self.bf = np.zeros(h, dtype=float)
        self.bi = np.zeros(h, dtype=float)
        self.bc = np.zeros(h, dtype=float)
        self.bo = np.zeros(h, dtype=float)
        self.Why = rng.normal(0, 0.1, (h, o)).astype(float)
        self.by = np.zeros(o, dtype=float)

    def _step(self, x, h, c):
        z = np.concatenate([x, h])
        f = sigmoid(z @ self.Wf + self.bf)
        i = sigmoid(z @ self.Wi + self.bi)
        c_hat = np.tanh(z @ self.Wc + self.bc)
        c = f * c + i * c_hat
        o = sigmoid(z @ self.Wo + self.bo)
        h = o * np.tanh(c)
        return h, c

    def _forward(self, X):
        X = np.asarray(X, dtype=float)
        B = len(X)
        hs = [np.zeros((B, self.hidden))]
        cs = [np.zeros((B, self.hidden))]
        for t in range(X.shape[1]):
            h, c = self._step(X[:, t], hs[-1], cs[-1])
            hs.append(h)
            cs.append(c)
        return hs, cs

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
                c = np.zeros(self.hidden)
                hs = [h.copy()]
                cs = [c.copy()]
                for t in range(T):
                    h, c = self._step(x[t], h, c)
                    hs.append(h.copy())
                    cs.append(c.copy())
                y_pred = h @ self.Why + self.by
                target = y[i] if y is not None else x[-1]
                loss = float(np.mean((y_pred - target) ** 2))
                epoch_loss += loss
                d_out = 2 * (y_pred - target) / max(target.size, 1)
                self.Why -= lr * np.outer(h, d_out.ravel())
                self.by -= lr * d_out.ravel()
                dh = (d_out @ self.Why.T).ravel()
                dc = np.zeros(self.hidden)
                for t in reversed(range(T)):
                    x_t = x[t]
                    h_prev = hs[t]
                    c_cur = cs[t + 1]
                    c_prev = cs[t]
                    z = np.concatenate([x_t, h_prev])
                    f = sigmoid(z @ self.Wf + self.bf)
                    i_gate = sigmoid(z @ self.Wi + self.bi)
                    c_hat = np.tanh(z @ self.Wc + self.bc)
                    o = sigmoid(z @ self.Wo + self.bo)
                    do = (dh * np.tanh(c_cur) * o * (1 - o)).ravel()
                    dc = (dh * o * (1 - np.tanh(c_cur) ** 2) + dc).ravel()
                    dc_hat = (dc * i_gate * (1 - c_hat ** 2)).ravel()
                    di = (dc * c_hat * i_gate * (1 - i_gate)).ravel()
                    df = (dc * c_prev * f * (1 - f)).ravel()
                    self.Wo -= lr * np.outer(z, do)
                    self.bo -= lr * do
                    self.Wi -= lr * np.outer(z, di)
                    self.bi -= lr * di
                    self.Wc -= lr * np.outer(z, dc_hat)
                    self.bc -= lr * dc_hat
                    self.Wf -= lr * np.outer(z, df)
                    self.bf -= lr * df
                    dh = (df @ self.Wf[self.input_dim:].T + di @ self.Wi[self.input_dim:].T +
                          dc_hat @ self.Wc[self.input_dim:].T + do @ self.Wo[self.input_dim:].T).ravel()
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
            c = np.zeros(self.hidden)
            for t in range(X.shape[1]):
                h, c = self._step(X[i, t], h, c)
            out.append((h @ self.Why + self.by).ravel())
        return np.array(out)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "Wf": self.Wf.tolist(), "Wi": self.Wi.tolist(),
            "Wc": self.Wc.tolist(), "Wo": self.Wo.tolist(),
            "bf": self.bf.tolist(), "bi": self.bi.tolist(),
            "bc": self.bc.tolist(), "bo": self.bo.tolist(),
            "Why": self.Why.tolist(), "by": self.by.tolist(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.Wf = np.asarray(state["Wf"], dtype=float)
        self.Wi = np.asarray(state["Wi"], dtype=float)
        self.Wc = np.asarray(state["Wc"], dtype=float)
        self.Wo = np.asarray(state["Wo"], dtype=float)
        self.bf = np.asarray(state["bf"], dtype=float)
        self.bi = np.asarray(state["bi"], dtype=float)
        self.bc = np.asarray(state["bc"], dtype=float)
        self.bo = np.asarray(state["bo"], dtype=float)
        self.Why = np.asarray(state["Why"], dtype=float)
        self.by = np.asarray(state["by"], dtype=float)
        self.trained = True
