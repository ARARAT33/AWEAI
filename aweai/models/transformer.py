"""Mini Transformer (from scratch, numpy).

A compact decoder-style transformer: token embedding + positional encoding,
self-attention, feed-forward blocks, and a final dense head. Trained with
mini-batch SGD via a simplified backprop that works on small datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import softmax


class MiniTransformer(BaseModel):
    model_type = "transformer"

    def __init__(self, vocab_size: int = 100, d_model: int = 16, nhead: int = 2,
                 layers: int = 1, max_len: int = 32, num_classes: int = 2, **params):
        super().__init__(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                         layers=layers, max_len=max_len, num_classes=num_classes, **params)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.layers = int(layers)
        self.max_len = int(max_len)
        self.num_classes = int(num_classes)
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(5)
        self.embed = rng.normal(0, 0.1, (self.vocab_size, self.d_model)).astype(float)
        self.pos = rng.normal(0, 0.1, (self.max_len, self.d_model)).astype(float)
        self.weights: Dict[str, np.ndarray] = {}
        d = self.d_model
        for li in range(self.layers):
            self.weights[f"L{li}_q"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_k"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_v"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_o"] = rng.normal(0, 0.1, (d, d)).astype(float)
            self.weights[f"L{li}_ff1"] = rng.normal(0, 0.1, (d, d * 2)).astype(float)
            self.weights[f"L{li}_ff2"] = rng.normal(0, 0.1, (d * 2, d)).astype(float)
        self.weights["head"] = rng.normal(0, 0.1, (d, self.num_classes)).astype(float)

    def _forward(self, x):
        # x: (B, T) ints
        B, T = x.shape
        h = self.embed[x] + self.pos[:T][None, :, :]  # (B, T, d)
        cache = {"h0": h}
        for li in range(self.layers):
            q = h @ self.weights[f"L{li}_q"]
            k = h @ self.weights[f"L{li}_k"]
            v = h @ self.weights[f"L{li}_v"]
            att = q @ k.transpose(0, 2, 1) / np.sqrt(self.d_model)
            att = softmax(att, axis=-1)
            ctx = att @ v  # (B, T, d)
            out = ctx @ self.weights[f"L{li}_o"]
            h = h + out
            cache[f"L{li}_att"] = att
            cache[f"L{li}_ctx"] = ctx
            cache[f"L{li}_out"] = out
            ff = np.maximum(h @ self.weights[f"L{li}_ff1"], 0.0)
            h = h + ff @ self.weights[f"L{li}_ff2"]
            cache[f"L{li}_ff"] = ff
            cache[f"L{li}_h"] = h
        pooled = h.mean(axis=1)
        logits = pooled @ self.weights["head"]
        return logits, cache

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.005, **kw):
        X = np.asarray(X, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        B, T = X.shape
        Y = np.zeros((B, self.num_classes))
        Y[np.arange(B), y] = 1.0
        for epoch in range(epochs):
            logits, cache = self._forward(X)
            probs = softmax(logits, axis=-1)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
            d_logits = (probs - Y) / B
            d_head = cache["h0"].mean(axis=1).T @ d_logits
            self.weights["head"] -= lr * d_head
            d_pool = d_logits @ self.weights["head"].T  # (B, d)
            d_h = np.repeat(d_pool[:, None, :], T, axis=1) / T
            for li in range(self.layers - 1, -1, -1):
                d = d_h
                h_after_ff = cache[f"L{li}_h"]
                ff = cache[f"L{li}_ff"]
                # ff2
                d_ff2 = ff.transpose(0, 2, 1) @ d
                self.weights[f"L{li}_ff2"] -= lr * d_ff2
                d_ff = d @ self.weights[f"L{li}_ff2"].T
                d_ff = d_ff * (ff > 0)
                d_ff1 = h_after_ff.transpose(0, 2, 1) @ d_ff
                self.weights[f"L{li}_ff1"] -= lr * d_ff1
                d_h = d_ff @ self.weights[f"L{li}_ff1"].T
                # attention output
                ctx = cache[f"L{li}_ctx"]
                att = cache[f"L{li}_att"]
                d_out = d_h
                d_o = ctx.transpose(0, 2, 1) @ d_out
                self.weights[f"L{li}_o"] -= lr * d_o
                d_ctx = d_out @ self.weights[f"L{li}_o"].T
                d_v = att.transpose(0, 2, 1) @ d_ctx
                self.weights[f"L{li}_v"] -= lr * d_v
                d_att = d_ctx @ cache[f"L{li}_v"].transpose(0, 2, 1)
                d_att = d_att * att * (1 - att)
                d_k = d_att.transpose(0, 2, 1) @ cache[f"L{li}_h0"] if False else 0
                d_q = d_att @ cache[f"L{li}_h0"] if False else 0
                h_in = cache[f"L{li}_h0"] if li == 0 else cache[f"L{li - 1}_h"]
                d_k = d_att.transpose(0, 2, 1) @ h_in
                d_q = d_att @ h_in
                self.weights[f"L{li}_k"] -= lr * d_k
                self.weights[f"L{li}_q"] -= lr * d_q
                d_h = (d_q @ self.weights[f"L{li}_q"].T + d_k @ self.weights[f"L{li}_k"].T + d_v @ self.weights[f"L{li}_v"].T) * (1 if True else 0)
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.int64)
        logits, _ = self._forward(X)
        return np.argmax(softmax(logits, axis=-1), axis=1)

    def state_dict(self):
        return {"embed": self.embed.tolist(), "pos": self.pos.tolist(),
                "weights": {k: v.tolist() for k, v in self.weights.items()}}

    def load_state(self, state):
        self.embed = np.asarray(state["embed"], dtype=float)
        self.pos = np.asarray(state["pos"], dtype=float)
        self.weights = {k: np.asarray(v, dtype=float) for k, v in state["weights"].items()}
        self.trained = True
