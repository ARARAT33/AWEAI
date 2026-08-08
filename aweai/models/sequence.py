"""Sequence models for time-series forecasting (v2.1).

* ``GRU``                 — gated recurrent unit (numpy, from scratch).
* ``TimeSeriesTransformer``— lightweight transformer for forecasting with
                            positional encoding + self-attention over a
                            lookback window (numpy, from scratch).

Both consume sequences shaped ``(N, T, input_dim)`` and predict the next
value(s) — regression-style forecasting — or classify a sequence when
``output_dim`` classes are provided. They implement the standard
``BaseModel`` interface so they integrate with the trainer/manager/exporter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid, softmax


class GRU(BaseModel):
    """Gated Recurrent Unit (from scratch, numpy).

    Standard update-gate / reset-gate formulation:

        z = sigmoid(x @ Wz + h @ Uz + bz)
        r = sigmoid(x @ Wr + h @ Ur + br)
        h_hat = tanh(x @ Wh + (r * h) @ Uh + bh)
        h = (1 - z) * h + z * h_hat

    ``fit`` supports teacher-forced forecasting: for each timestep t it
    predicts the next timestep's value using the current hidden state, then
    trains by MSE against ``x[t+1]`` (or provided ``y``).
    """

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
        rng = np.random.default_rng(3)
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
                hs = [h]
                for t in range(len(x)):
                    h = self._step(x[t], h)
                    hs.append(h)
                y_pred = h @ self.Why + self.by
                y_pred = np.asarray(y_pred, dtype=float).reshape(1, -1)[:, : self.output_dim]
                if y is not None:
                    target = np.asarray(y[i], dtype=float).reshape(1, -1)[:, : self.output_dim]
                else:
                    target = np.roll(x, -1, axis=0)[-1:].reshape(1, -1)[:, : self.output_dim]
                    if np.allclose(target, 0) and len(x) > 1:
                        target = x[-2:-1, : self.output_dim]
                loss = np.mean((y_pred - target) ** 2)
                epoch_loss += loss
                d_out = 2 * (y_pred - target) / max(target.size, 1)
                self.Why -= lr * np.outer(h, d_out.ravel())
                self.by -= lr * d_out.ravel()
                dh = (d_out @ self.Why.T).ravel()
                # backprop through GRU steps
                for t in reversed(range(len(x))):
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
                    dh = (dz @ self.Uz.T + dr @ self.Ur.T +
                          (dhh @ self.Uh.T) * r).ravel()
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
                h = self._step(x[t], h)
            out.append((h @ self.Why + self.by).ravel())
        return np.array(out)

    def forecast(self, X, steps: int = 1):
        """Iterative multi-step forecasting."""
        X = np.asarray(X, dtype=float)
        seq = X[0].copy() if X.ndim == 2 else X
        if seq.ndim == 1:
            seq = seq.reshape(-1, self.input_dim)
        out = []
        for _ in range(int(steps)):
            h = np.zeros(self.hidden)
            for t in range(len(seq)):
                h = self._step(seq[t], h)
            nxt = (h @ self.Why + self.by).ravel()
            out.append(nxt)
            seq = np.vstack([seq, nxt[: self.input_dim]])
        return np.array(out)

    def state_dict(self) -> Dict[str, Any]:
        return {"Wz": self.Wz.tolist(), "Uz": self.Uz.tolist(), "bz": self.bz.tolist(),
                "Wr": self.Wr.tolist(), "Ur": self.Ur.tolist(), "br": self.br.tolist(),
                "Wh": self.Wh.tolist(), "Uh": self.Uh.tolist(), "bh": self.bh.tolist(),
                "Why": self.Why.tolist(), "by": self.by.tolist()}

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


class TimeSeriesTransformer(BaseModel):
    """Lightweight transformer for time-series forecasting (numpy).

    Encodes the lookback window with learnable per-timestep positional
    encodings, applies ``layers`` self-attention + feed-forward blocks, then
    pools and predicts the next value(s). Trained by MSE with a compact
    backprop that mirrors the factory's ``MiniTransformer``.
    """

    model_type = "ts_transformer"
    is_regressor = True

    def __init__(self, input_dim: int = 1, d_model: int = 16, nhead: int = 2,
                 layers: int = 1, max_len: int = 32, output_dim: int = 1, **params):
        super().__init__(input_dim=input_dim, d_model=d_model, nhead=nhead,
                         layers=layers, max_len=max_len, output_dim=output_dim, **params)
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.layers = int(layers)
        self.max_len = int(max_len)
        self.output_dim = int(output_dim)
        self._build()

    def _build(self):
        rng = np.random.default_rng(9)
        d = self.d_model
        self.input_proj = rng.normal(0, 0.1, (self.input_dim, d)).astype(float)
        self.pos = rng.normal(0, 0.1, (self.max_len, d)).astype(float)
        self.weights: Dict[str, np.ndarray] = {}
        for li in range(self.layers):
            self.weights[f"L{li}_q"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_k"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_v"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_o"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_ff1"] = rng.normal(0, 0.1, (d, d * 2)).astype(float)
            self.weights[f"L{li}_ff2"] = rng.normal(0, 0.1, (d * 2, d)).astype(float)
        self.weights["head"] = rng.normal(0, 0.1, (d, self.output_dim)).astype(float)

    def _forward(self, x):
        # x: (B, T, input_dim)
        B, T, _ = x.shape
        h = x @ self.input_proj + self.pos[:T][None, :, :]
        cache = {"h0": h}
        for li in range(self.layers):
            q = h @ self.weights[f"L{li}_q"]
            k = h @ self.weights[f"L{li}_k"]
            v = h @ self.weights[f"L{li}_v"]
            att = q @ k.transpose(0, 2, 1) / np.sqrt(self.d_model)
            att = softmax(att, axis=-1)
            ctx = att @ v
            out = ctx @ self.weights[f"L{li}_o"]
            h = h + out
            cache[f"L{li}_att"] = att
            cache[f"L{li}_ctx"] = ctx
            cache[f"L{li}_out"] = out
            cache[f"L{li}_v"] = v
            ff = np.maximum(h @ self.weights[f"L{li}_ff1"], 0.0)
            h = h + ff @ self.weights[f"L{li}_ff2"]
            cache[f"L{li}_ff"] = ff
            cache[f"L{li}_h"] = h
        pooled = h.mean(axis=1)
        pred = pooled @ self.weights["head"]
        return pred, cache

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.005, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), -1, self.input_dim)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        B, T, D = X.shape
        if y is None:
            # next-step prediction target: shift left
            y = np.zeros((B, self.output_dim))
            for i in range(B):
                y[i] = X[i, -1, : self.output_dim]
        Y = np.asarray(y, dtype=float).reshape(B, -1)[:, : self.output_dim]
        for epoch in range(epochs):
            pred, cache = self._forward(X)
            loss = np.mean((pred - Y) ** 2)
            d = 2 * (pred - Y) / max(Y.size, 1)
            self.weights["head"] -= lr * (cache["h0"].mean(axis=1).T @ d)
            d_pool = d @ self.weights["head"].T
            d_h = np.repeat(d_pool[:, None, :], T, axis=1) / T
            for li in range(self.layers - 1, -1, -1):
                dd = d_h
                h_after_ff = cache[f"L{li}_h"]
                ff = cache[f"L{li}_ff"]
                self.weights[f"L{li}_ff2"] -= lr * np.mean(ff.transpose(0, 2, 1) @ dd, axis=0)
                d_ff = dd @ self.weights[f"L{li}_ff2"].T
                d_ff = d_ff * (ff > 0)
                self.weights[f"L{li}_ff1"] -= lr * np.mean(h_after_ff.transpose(0, 2, 1) @ d_ff, axis=0)
                d_h = d_ff @ self.weights[f"L{li}_ff1"].T
                ctx = cache[f"L{li}_ctx"]
                att = cache[f"L{li}_att"]
                d_out = d_h
                self.weights[f"L{li}_o"] -= lr * np.mean(ctx.transpose(0, 2, 1) @ d_out, axis=0)
                d_ctx = d_out @ self.weights[f"L{li}_o"].T
                h_in = cache["h0"] if li == 0 else cache[f"L{li - 1}_h"]
                self.weights[f"L{li}_v"] -= lr * np.mean(h_in.transpose(0, 2, 1) @ d_ctx, axis=0)
                d_att = d_ctx @ cache[f"L{li}_v"].transpose(0, 2, 1)
                d_att = d_att * att * (1 - att)
                d_k = d_att.transpose(0, 2, 1) @ h_in
                d_q = d_att @ h_in
                self.weights[f"L{li}_k"] -= lr * np.mean(d_k, axis=0)
                self.weights[f"L{li}_q"] -= lr * np.mean(d_q, axis=0)
                d_h = (d_q @ self.weights[f"L{li}_q"].T +
                       d_k @ self.weights[f"L{li}_k"].T +
                       d_ctx @ self.weights[f"L{li}_v"].T)
            # input projection gradient (simple mean approximation)
            self.input_proj -= lr * np.mean(np.einsum("bti,btd->bid", X, d_h), axis=0)
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), -1, self.input_dim)
        pred, _ = self._forward(X)
        return pred

    def forecast(self, X, steps: int = 1):
        """Iterative multi-step forecast (auto-regressive)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), -1, self.input_dim)
        seq = X[0].copy()
        out = []
        for _ in range(int(steps)):
            pred, _ = self._forward(seq[None, :, :])
            nxt = pred[0]
            out.append(nxt)
            seq = np.vstack([seq[1:], nxt[: self.input_dim]])
        return np.array(out)

    def state_dict(self) -> Dict[str, Any]:
        return {"input_proj": self.input_proj.tolist(), "pos": self.pos.tolist(),
                "weights": {k: v.tolist() for k, v in self.weights.items()}}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.input_proj = np.asarray(state["input_proj"], dtype=float)
        self.pos = np.asarray(state["pos"], dtype=float)
        self.weights = {k: np.asarray(v, dtype=float) for k, v in state["weights"].items()}
        self.trained = True
