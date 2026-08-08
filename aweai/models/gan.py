"""Generative Adversarial Network (GAN) — MLP-based, trained in numpy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


class GAN(BaseModel):
    model_type = "gan"
    is_generative = True

    def __init__(self, input_dim: int = 2, latent: int = 4, hidden: Optional[List[int]] = None, **params):
        super().__init__(input_dim=input_dim, latent=latent, hidden=hidden, **params)
        self.input_dim = int(input_dim)
        self.latent = int(latent)
        self.hidden = [int(h) for h in (hidden or [8, 8])]
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(3)
        g_sizes = [self.latent] + self.hidden + [self.input_dim]
        d_sizes = [self.input_dim] + self.hidden[::-1] + [1]
        self.GW = [rng.normal(0, 0.1, (g_sizes[i], g_sizes[i + 1])).astype(float) for i in range(len(g_sizes) - 1)]
        self.Gb = [np.zeros(s, dtype=float) for s in g_sizes[1:]]
        self.DW = [rng.normal(0, 0.1, (d_sizes[i], d_sizes[i + 1])).astype(float) for i in range(len(d_sizes) - 1)]
        self.Db = [np.zeros(s, dtype=float) for s in d_sizes[1:]]

    def _gen(self, z):
        a = z
        for i in range(len(self.GW) - 1):
            a = np.tanh(a @ self.GW[i] + self.Gb[i])
        return a @ self.GW[-1] + self.Gb[-1]

    def _disc(self, x):
        a = x
        for i in range(len(self.DW) - 1):
            a = np.tanh(a @ self.DW[i] + self.Db[i])
        return a @ self.DW[-1] + self.Db[-1]

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.001, batch_size: int = 16, **kw):
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        batch_size = int(kw.get("batch_size", batch_size))
        n = len(X)
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                real = X[idx]
                z = np.random.randn(len(real), self.latent).astype(float)
                fake = self._gen(z)
                d_real = self._disc(real)
                d_fake = self._disc(fake)
                d_loss = -np.mean(np.log(1e-12 + 1 / (1 + np.exp(-d_real))) + np.log(1e-12 + 1 - 1 / (1 + np.exp(-d_fake))))
                # simpler BCE
                p_real = 1 / (1 + np.exp(-d_real))
                p_fake = 1 / (1 + np.exp(-d_fake))
                d_loss = -np.mean(np.log(p_real + 1e-12) + np.log(1 - p_fake + 1e-12))
                g_loss = -np.mean(np.log(p_fake + 1e-12))
                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                # update discriminator (gradient ascent on -loss -> descent)
                d_out_real = (p_real - 1) / len(real)
                d_out_fake = p_fake / len(real)
                # combine: real grad + fake grad w.r.t. d_fake input
                self._update_disc(real, d_out_real, lr)
                self._update_disc(fake, d_out_fake, lr)
                # update generator
                self._update_gen(z, p_fake, lr)
            self.history["loss"].append(float(np.mean(d_losses)))
            self.history["val_loss"].append(float(np.mean(g_losses)))
        self.trained = True
        self.metrics["d_loss"] = float(np.mean(d_losses))
        self.metrics["g_loss"] = float(np.mean(g_losses))
        return self

    def _update_disc(self, x, grad, lr):
        # backprop through discriminator
        acts = [x]
        for i in range(len(self.DW) - 1):
            a = np.tanh(acts[-1] @ self.DW[i] + self.Db[i])
            acts.append(a)
        z = acts[-1] @ self.DW[-1] + self.Db[-1]
        p = 1 / (1 + np.exp(-z))
        d = grad * (p * (1 - p))
        d = d.reshape(-1, 1)
        grads_w = [None] * len(self.DW)
        grads_b = [None] * len(self.Db)
        for i in range(len(self.DW) - 1, -1, -1):
            grads_w[i] = acts[i].T @ d
            grads_b[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.DW[i].T) * (1 - acts[i] ** 2)
        for i in range(len(self.DW)):
            self.DW[i] -= lr * grads_w[i]
            self.Db[i] -= lr * grads_b[i]

    def _update_gen(self, z, p_fake, lr):
        g_acts = [z]
        for i in range(len(self.GW) - 1):
            a = np.tanh(g_acts[-1] @ self.GW[i] + self.Gb[i])
            g_acts.append(a)
        fake = g_acts[-1] @ self.GW[-1] + self.Gb[-1]
        d = self._disc(fake)
        p = 1 / (1 + np.exp(-d))
        # d loss w.r.t. fake: -1/(1-p) * p*(1-p) = -p
        grad_fake = -p.reshape(-1, 1)
        d_acts = [fake]
        for i in range(len(self.DW) - 1):
            a = np.tanh(d_acts[-1] @ self.DW[i] + self.Db[i])
            d_acts.append(a)
        zz = d_acts[-1] @ self.DW[-1] + self.Db[-1]
        pp = 1 / (1 + np.exp(-zz))
        grad = grad_fake * (pp * (1 - pp))
        for i in range(len(self.DW) - 1, 0, -1):
            grad = (grad @ self.DW[i].T) * (1 - d_acts[i] ** 2)
        grad = grad @ self.DW[0].T
        # backprop through generator
        g_grads_w = [None] * len(self.GW)
        g_grads_b = [None] * len(self.Gb)
        g = grad
        for i in range(len(self.GW) - 1, -1, -1):
            g_grads_w[i] = g_acts[i].T @ g
            g_grads_b[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.GW[i].T) * (1 - g_acts[i] ** 2)
        for i in range(len(self.GW)):
            self.GW[i] -= lr * g_grads_w[i]
            self.Gb[i] -= lr * g_grads_b[i]

    def generate(self, n: int = 5, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, (n, self.latent)).astype(float)
        return self._gen(z)

    def predict(self, X):
        return self.generate(n=len(np.asarray(X)))

    def state_dict(self):
        return {
            "GW": [w.tolist() for w in self.GW],
            "Gb": [b.tolist() for b in self.Gb],
            "DW": [w.tolist() for w in self.DW],
            "Db": [b.tolist() for b in self.Db],
        }

    def load_state(self, state):
        self.GW = [np.asarray(w, dtype=float) for w in state["GW"]]
        self.Gb = [np.asarray(b, dtype=float) for b in state["Gb"]]
        self.DW = [np.asarray(w, dtype=float) for w in state["DW"]]
        self.Db = [np.asarray(b, dtype=float) for b in state["Db"]]
        self.trained = True
