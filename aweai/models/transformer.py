"""Mini Transformer implemented from scratch in numpy.

A compact encoder-only transformer for text classification tasks.
Supports multiple attention heads, feed-forward blocks and a learnable
token embedding. Trained with softmax cross-entropy and a compact
backprop that mirrors the factory's other numpy models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import softmax


class MiniTransformer(BaseModel):
    model_type = "transformer"

    def __init__(self, vocab_size: int = 100, d_model: int = 16, nhead: int = 2,
                 layers: int = 1, num_classes: int = 2, **params):
        super().__init__(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                         layers=layers, num_classes=num_classes, **params)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.layers = int(layers)
        self.num_classes = int(num_classes)
        self._build()

    def _build(self):
        rng = np.random.default_rng(11)
        self.embed = rng.normal(0, 0.1, (self.vocab_size, self.d_model)).astype(float)
        self.pos = rng.normal(0, 0.1, (64, self.d_model)).astype(float)
        self.weights: Dict[str, np.ndarray] = {}
        for li in range(self.layers):
            self.weights[f"L{li}_q"] = rng.normal(0, 0.1, (self.d_model, self.d_model)).astype(float)
            self.weights[f"L{li}_k"] = rng.normal(0, 0.1, (self.d_model, self.d_model)).astype(float)
            self.weights[f"L{li}_v"] = rng.normal(0, 0.1, (self.d_model, self.d_model)).astype(float)
            self.weights[f"L{li}_o"] = rng.normal(0, 0.1, (self.d_model, self.d_model)).astype(float)
            self.weights[f"L{li}_ff1"] = rng.normal(0, 0.1, (self.d_model, self.d_model * 2)).astype(float)
            self.weights[f"L{li}_ff2"] = rng.normal(0, 0.1, (self.d_model * 2, self.d_model)).astype(float)
        self.weights["head"] = rng.normal(0, 0.1, (self.d_model, self.num_classes)).astype(float)

    def _forward(self, x):
        # x: (B, T) token ids
        B, T = x.shape
        h = self.embed[x] + self.pos[:T][None, :, :]
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
        logits = pooled @ self.weights["head"]
        return logits, cache

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.005, **kw):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        B, T = X.shape
        Y = np.zeros((B, self.num_classes))
        Y[np.arange(B), y] = 1.0
        for epoch in range(epochs):
            logits, cache = self._forward(X)
            probs = softmax(logits, axis=-1)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
            d = (probs - Y) / B
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
                h = cache["h0"] if li == 0 else cache[f"L{li - 1}_h"]
                q = h @ self.weights[f"L{li}_q"]
                k = h @ self.weights[f"L{li}_k"]
                d_q = d_att @ k
                d_k = d_att.transpose(0, 2, 1) @ q
                self.weights[f"L{li}_q"] -= lr * np.einsum("bti,btj->ij", h, d_q) / B
                self.weights[f"L{li}_k"] -= lr * np.einsum("bti,btj->ij", h, d_k) / B
                d_h = (d_q @ self.weights[f"L{li}_q"].T + d_k @ self.weights[f"L{li}_k"].T + d_ctx @ self.weights[f"L{li}_v"].T)
            # embed + pos gradients (simple approximation)
            d_h0 = d_h
            grad_embed = np.zeros_like(self.embed)
            np.add.at(grad_embed, X, d_h0)
            self.embed -= lr * grad_embed / B
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=int)
        logits, _ = self._forward(X)
        return np.argmax(softmax(logits, axis=-1), axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=int)
        logits, _ = self._forward(X)
        return softmax(logits, axis=-1)

    def state_dict(self) -> Dict[str, Any]:
        return {"embed": self.embed.tolist(), "pos": self.pos.tolist(),
                "weights": {k: v.tolist() for k, v in self.weights.items()}}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.embed = np.asarray(state["embed"], dtype=float)
        self.pos = np.asarray(state["pos"], dtype=float)
        self.weights = {k: np.asarray(v, dtype=float) for k, v in state["weights"].items()}
        self.trained = True
