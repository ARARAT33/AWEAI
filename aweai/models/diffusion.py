"""Denoising Diffusion Probabilistic Model (DDPM) from scratch in numpy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


class DiffusionModel(BaseModel):
    model_type = "diffusion"
    is_generative = True

    def __init__(self, input_dim: int = 2, hidden: Optional[List[int]] = None,
                 timesteps: int = 100, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, timesteps=timesteps, **params)
        self.input_dim = int(input_dim)
        self.hidden = [int(h) for h in (hidden or [16, 16])]
        self.timesteps = int(timesteps)
        self._build()

    def _build(self):
        rng = np.random.default_rng(33)
        sizes = [self.input_dim + 1] + self.hidden + [self.input_dim]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, 0.1, (sizes[i], sizes[i + 1])).astype(float))
            self.b.append(np.zeros(sizes[i + 1], dtype=float))
        self.beta = np.linspace(1e-4, 0.02, self.timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def _forward(self, X, t):
        t_norm = np.full((len(X), 1), t / self.timesteps, dtype=float)
        acts = [np.hstack([X, t_norm])]
        for i in range(len(self.W) - 1):
            a = np.maximum(acts[-1] @ self.W[i] + self.b[i], 0.0)
            acts.append(a)
        out = acts[-1] @ self.W[-1] + self.b[-1]
        acts.append(out)
        return acts

    def _diffuse(self, X, t):
        noise = np.random.randn(*X.shape)
        sqrt_ab = np.sqrt(self.alpha_bar[t])
        sqrt_1m = np.sqrt(1 - self.alpha_bar[t])
        return sqrt_ab * X + sqrt_1m * noise, noise

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.005, **kw):
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        n = len(X)
        for _ in range(epochs):
            epoch_loss = 0.0
            for _ in range(n):
                t = np.random.randint(1, self.timesteps)
                x0 = X[np.random.randint(n)]
                x_t, noise = self._diffuse(x0, t)
                acts = self._forward(x_t[None, :], t)
                pred = acts[-1]
                loss = float(np.mean((pred - noise) ** 2))
                epoch_loss += loss
                d = 2 * (pred - noise) / noise.size
                for i in range(len(self.W) - 1, -1, -1):
                    a_prev = acts[i]
                    self.W[i] -= lr * (a_prev.T @ d)
                    self.b[i] -= lr * d.sum(axis=0)
                    if i > 0:
                        d = (d @ self.W[i].T) * (a_prev > 0)
            self.history["loss"].append(float(epoch_loss / n))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def sample(self, n: int = 5, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.input_dim)).astype(float)
        for t in reversed(range(1, self.timesteps)):
            pred = self._forward(x, t)[-1]
            z = rng.standard_normal(x.shape) if t > 1 else np.zeros_like(x)
            x = (x - (self.beta[t] / (1 - self.alpha_bar[t])) * pred) / np.sqrt(self.alpha[t])
            x = x + np.sqrt(self.beta[t]) * z
        return x

    def predict(self, X):
        return self.sample(n=len(np.asarray(X)))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "W": [w.tolist() for w in self.W],
            "b": [b.tolist() for b in self.b],
            "timesteps": self.timesteps,
            "beta": self.beta.tolist(),
            "alpha": self.alpha.tolist(),
            "alpha_bar": self.alpha_bar.tolist(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.W = [np.asarray(w, dtype=float) for w in state["W"]]
        self.b = [np.asarray(b, dtype=float) for b in state["b"]]
        self.timesteps = int(state.get("timesteps", self.timesteps))
        self.beta = np.asarray(state.get("beta", self.beta), dtype=float)
        self.alpha = np.asarray(state.get("alpha", self.alpha), dtype=float)
        self.alpha_bar = np.asarray(state.get("alpha_bar", self.alpha_bar), dtype=float)
        self.trained = True
