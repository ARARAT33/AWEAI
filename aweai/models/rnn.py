"""Simple RNN and LSTM implemented from scratch in numpy.

Used for text / time-series tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid, softmax


class RNN(BaseModel):
    model_type = "rnn"

    def __init__(self, input_dim: int = 1, hidden: int = 16, output_dim: int = 1,
                 layers: int = 1, seq_len: int = 4, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, output_dim=output_dim,
                         layers=layers, seq_len=seq_len, **params)
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.output_dim = int(output_dim)
        self.layers = int(layers)
        self.seq_len = int(seq_len)
        self.Wxh = np.random.randn(self.input_dim, self.hidden) * 0.1
        self.Whh = np.random.randn(self.hidden, self.hidden) * 0.1
        self.bh = np.zeros(self.hidden)
        self.Why = np.random.randn(self.hidden, self.output_dim) * 0.1
        self.by = np.zeros(self.output_dim)

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.01, **kw):
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        n = len(X)
        loss_total = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(n):
                x = X[i]
                if x.ndim == 1:
                    x = x.reshape(-1, self.input_dim)
                h = np.zeros(self.hidden)
                hs = [h]
                for t in range(len(x)):
                    h = np.tanh(x[t] @ self.Wxh + h @ self.Whh + self.bh)
                    hs.append(h)
                y_pred = hs[-1] @ self.Why + self.by
                y_pred = np.asarray(y_pred, dtype=float).reshape(1, -1)[:, : self.output_dim]
                target = np.asarray(y[i], dtype=float).reshape(1, -1)[:, : self.output_dim] if y is not None else y_pred
                if y is None:
                    target = np.roll(x, -1, axis=0)[-1:].reshape(1, -1) if x.shape[0] > 1 else np.zeros((1, self.output_dim))
                    target = target[:, : self.output_dim]
                loss = np.mean((y_pred - target) ** 2)
                epoch_loss += loss
                d_out = 2 * (y_pred - target) / target.size
                dWhy = hs[-1].reshape(-1, 1) @ d_out.reshape(1, -1)
                dby = d_out.ravel()
                dh = (d_out @ self.Why.T).ravel()
                dWxh = np.zeros_like(self.Wxh)
                dWhh = np.zeros_like(self.Whh)
                dbh = np.zeros_like(self.bh)
                for t in reversed(range(len(x))):
                    dtanh = dh * (1 - hs[t + 1] ** 2)
                    dWxh += np.outer(x[t], dtanh)
                    dWhh += np.outer(hs[t], dtanh)
                    dbh += dtanh
                    dh = dtanh @ self.Whh.T
                self.Wxh -= lr * dWxh
                self.Whh -= lr * dWhh
                self.bh -= lr * dbh
                self.Why -= lr * dWhy
                self.by -= lr * dby
            self.history["loss"].append(float(epoch_loss / max(n, 1)))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for i in range(len(X)):
            x = X[i]
            if x.ndim == 1:
                x = x.reshape(-1, self.input_dim)
            h = np.zeros(self.hidden)
            for t in range(len(x)):
                h = np.tanh(x[t] @ self.Wxh + h @ self.Whh + self.bh)
            out.append((h @ self.Why + self.by).ravel())
        return np.array(out)

    def state_dict(self):
        return {"Wxh": self.Wxh.tolist(), "Whh": self.Whh.tolist(), "bh": self.bh.tolist(),
                "Why": self.Why.tolist(), "by": self.by.tolist()}

    def load_state(self, state):
        self.Wxh = np.asarray(state["Wxh"], dtype=float)
        self.Whh = np.asarray(state["Whh"], dtype=float)
        self.bh = np.asarray(state["bh"], dtype=float)
        self.Why = np.asarray(state["Why"], dtype=float)
        self.by = np.asarray(state["by"], dtype=float)
        self.trained = True


class LSTM(RNN):
    """Minimal LSTM: one hidden layer, MSE training for sequences."""

    model_type = "lstm"

    def __init__(self, input_dim: int = 1, hidden: int = 16, output_dim: int = 1, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, output_dim=output_dim, **params)
        self.Wf = np.random.randn(input_dim + hidden, hidden) * 0.1
        self.Wi = np.random.randn(input_dim + hidden, hidden) * 0.1
        self.Wc = np.random.randn(input_dim + hidden, hidden) * 0.1
        self.Wo = np.random.randn(input_dim + hidden, hidden) * 0.1
        self.bf = np.zeros(hidden)
        self.bi = np.zeros(hidden)
        self.bc = np.zeros(hidden)
        self.bo = np.zeros(hidden)
        self.Why = np.random.randn(hidden, output_dim) * 0.1
        self.by = np.zeros(output_dim)

    def _step(self, x, h, c):
        z = np.concatenate([x, h])
        f = sigmoid(z @ self.Wf + self.bf)
        i = sigmoid(z @ self.Wi + self.bi)
        c_hat = np.tanh(z @ self.Wc + self.bc)
        c = f * c + i * c_hat
        o = sigmoid(z @ self.Wo + self.bo)
        h = o * np.tanh(c)
        return h, c

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.01, **kw):
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        n = len(X)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(n):
                x = X[i]
                if x.ndim == 1:
                    x = x.reshape(-1, self.input_dim)
                h = np.zeros(self.hidden)
                c = np.zeros(self.hidden)
                hs = []
                cs = []
                for t in range(len(x)):
                    h, c = self._step(x[t], h, c)
                    hs.append(h)
                    cs.append(c)
                y_pred = h @ self.Why + self.by
                y_pred = np.asarray(y_pred, dtype=float).reshape(1, -1)[:, : self.output_dim]
                if y is None:
                    target = np.roll(x, -1, axis=0)[-1:].reshape(1, -1)[:, : self.output_dim]
                else:
                    target = np.asarray(y[i], dtype=float).reshape(1, -1)[:, : self.output_dim]
                loss = np.mean((y_pred - target) ** 2)
                epoch_loss += loss
                d_out = 2 * (y_pred - target) / target.size
                self.Why -= lr * np.outer(h, d_out.ravel())
                self.by -= lr * d_out.ravel()
                dh = (d_out @ self.Why.T).ravel()
                dc = np.zeros(self.hidden)
                for t in reversed(range(len(x))):
                    o = sigmoid(np.concatenate([x[t], hs[t - 1] if t > 0 else np.zeros(self.hidden)]) @ self.Wo + self.bo)
                    c_cur = cs[t]
                    c_prev = cs[t - 1] if t > 0 else np.zeros(self.hidden)
                    z = np.concatenate([x[t], hs[t - 1] if t > 0 else np.zeros(self.hidden)])
                    f = sigmoid(z @ self.Wf + self.bf)
                    i = sigmoid(z @ self.Wi + self.bi)
                    c_hat = np.tanh(z @ self.Wc + self.bc)
                    do = (dh * np.tanh(c_cur) * o * (1 - o)).ravel()
                    dc = (dh * o * (1 - np.tanh(c_cur) ** 2) + dc).ravel()
                    dc_hat = (dc * i * (1 - c_hat ** 2)).ravel()
                    di = (dc * c_hat * i * (1 - i)).ravel()
                    df = (dc * c_prev * f * (1 - f)).ravel()
                    self.Wo -= lr * np.outer(z, do)
                    self.bo -= lr * do
                    self.Wi -= lr * np.outer(z, di)
                    self.bi -= lr * di
                    self.Wc -= lr * np.outer(z, dc_hat)
                    self.bc -= lr * dc_hat
                    self.Wf -= lr * np.outer(z, df)
                    self.bf -= lr * df
                    dh = (df @ self.Wf[: self.input_dim].T + di @ self.Wi[: self.input_dim].T +
                          dc_hat @ self.Wc[: self.input_dim].T + do @ self.Wo[: self.input_dim].T).ravel()
            self.history["loss"].append(float(epoch_loss / max(n, 1)))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for i in range(len(X)):
            x = X[i]
            if x.ndim == 1:
                x = x.reshape(-1, self.input_dim)
            h = np.zeros(self.hidden)
            c = np.zeros(self.hidden)
            for t in range(len(x)):
                h, c = self._step(x[t], h, c)
            out.append((h @ self.Why + self.by).ravel())
        return np.array(out)

    def state_dict(self):
        return {"Wf": self.Wf.tolist(), "Wi": self.Wi.tolist(), "Wc": self.Wc.tolist(), "Wo": self.Wo.tolist(),
                "bf": self.bf.tolist(), "bi": self.bi.tolist(), "bc": self.bc.tolist(), "bo": self.bo.tolist(),
                "Why": self.Why.tolist(), "by": self.by.tolist()}

    def load_state(self, state):
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
