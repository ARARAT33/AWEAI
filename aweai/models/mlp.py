"""Multi-layer perceptron classifier/regressor trained with SGD + backprop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import sigmoid, softmax


class MLP(BaseModel):
    """Fully-connected network, trained with mini-batch SGD + backprop.

    Supports classification (softmax cross-entropy) and regression (MSE).
    Hidden sizes come from params['hidden'] (list of ints).
    """

    model_type = "mlp"

    def __init__(self, input_dim: int = 2, output_dim: int = 1, hidden: Optional[List[int]] = None, **params):
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden=hidden, **params)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = [int(h) for h in (hidden or [8])]
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(42)
        sizes = [self.input_dim] + self.hidden + [self.output_dim]
        self.Ws: List[np.ndarray] = []
        self.bs: List[np.ndarray] = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.Ws.append(rng.normal(0.0, scale, (sizes[i], sizes[i + 1])).astype(float))
            self.bs.append(np.zeros(sizes[i + 1], dtype=float))

    def _forward(self, X: np.ndarray):
        a = np.asarray(X, dtype=float)
        acts = [a]
        for i in range(len(self.Ws) - 1):
            z = acts[-1] @ self.Ws[i] + self.bs[i]
            a = np.maximum(z, 0.0)  # relu
            acts.append(a)
        z_out = acts[-1] @ self.Ws[-1] + self.bs[-1]
        acts.append(z_out)
        return acts

    def fit(self, X, y=None, epochs: int = 50, lr: float = 0.01, batch_size: int = 16,
            val_X=None, val_y=None, early_stopping: bool = False, patience: int = 5, **kw):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        batch_size = int(kw.get("batch_size", batch_size))
        n = len(X)
        if self.output_dim == 1 and y.ndim == 1:
            y = y.reshape(-1, 1)
        classification = self.is_classifier or (self.output_dim > 1 and y.ndim == 2 and y.shape[1] > 1)
        best_loss = float("inf")
        patience_count = 0
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X[idx]
                yb = y[idx]
                acts = self._forward(xb)
                z_out = acts[-1]
                if classification:
                    probs = softmax(z_out)
                    loss = -np.mean(np.sum(yb * np.log(probs + 1e-12), axis=1))
                    d_out = (probs - yb) / len(xb)
                else:
                    loss = np.mean((z_out - yb) ** 2)
                    d_out = 2 * (z_out - yb) / len(xb)
                epoch_loss += loss * len(xb)
                # backprop
                grads_w = [None] * len(self.Ws)
                grads_b = [None] * len(self.bs)
                d = d_out
                for i in range(len(self.Ws) - 1, -1, -1):
                    a_prev = acts[i]
                    grads_w[i] = a_prev.T @ d
                    grads_b[i] = d.sum(axis=0)
                    if i > 0:
                        d = (d @ self.Ws[i].T) * (acts[i] > 0)
                for i in range(len(self.Ws)):
                    self.Ws[i] -= lr * grads_w[i]
                    self.bs[i] -= lr * grads_b[i]
            epoch_loss /= n
            self.history["loss"].append(float(epoch_loss))
            if val_X is not None:
                vpred = self.predict(val_X)
                vloss = float(np.mean((vpred - np.asarray(val_y)) ** 2))
                self.history["val_loss"].append(vloss)
                if early_stopping and vloss < best_loss - 1e-5:
                    best_loss = vloss
                    patience_count = 0
                elif early_stopping:
                    patience_count += 1
                    if patience_count >= patience:
                        break
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1]) if self.history["loss"] else 0.0
        return self

    def predict(self, X):
        acts = self._forward(X)
        z_out = acts[-1]
        if self.output_dim == 1:
            return z_out.reshape(-1)
        return np.argmax(softmax(z_out), axis=1)

    def predict_proba(self, X):
        return softmax(self._forward(X)[-1])

    def state_dict(self) -> Dict[str, Any]:
        return {"Ws": [w.tolist() for w in self.Ws], "bs": [b.tolist() for b in self.bs]}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.Ws = [np.asarray(w, dtype=float) for w in state["Ws"]]
        self.bs = [np.asarray(b, dtype=float) for b in state["bs"]]
        self.trained = True
