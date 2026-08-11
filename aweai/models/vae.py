"""Variational Autoencoder (VAE) implemented from scratch in numpy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


class VAE(BaseModel):
    model_type = "vae"
    is_generative = True

    def __init__(self, input_dim: int = 2, hidden: Optional[List[int]] = None,
                 latent_dim: int = 2, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, latent_dim=latent_dim, **params)
        self.input_dim = int(input_dim)
        self.hidden = [int(h) for h in (hidden or [8, 4])]
        self.latent_dim = int(latent_dim)
        self._build()

    def _build(self):
        rng = np.random.default_rng(21)
        sizes = [self.input_dim] + self.hidden
        self.encoder_W: List[np.ndarray] = []
        self.encoder_b: List[np.ndarray] = []
        for i in range(len(sizes) - 1):
            self.encoder_W.append(rng.normal(0, 0.1, (sizes[i], sizes[i + 1])).astype(float))
            self.encoder_b.append(np.zeros(sizes[i + 1], dtype=float))
        self.mu_W = rng.normal(0, 0.1, (sizes[-1], self.latent_dim)).astype(float)
        self.mu_b = np.zeros(self.latent_dim, dtype=float)
        self.logvar_W = rng.normal(0, 0.1, (sizes[-1], self.latent_dim)).astype(float)
        self.logvar_b = np.zeros(self.latent_dim, dtype=float)
        dec_sizes = [self.latent_dim] + self.hidden[::-1] + [self.input_dim]
        self.decoder_W: List[np.ndarray] = []
        self.decoder_b: List[np.ndarray] = []
        for i in range(len(dec_sizes) - 1):
            self.decoder_W.append(rng.normal(0, 0.1, (dec_sizes[i], dec_sizes[i + 1])).astype(float))
            self.decoder_b.append(np.zeros(dec_sizes[i + 1], dtype=float))

    def _encode(self, X):
        a = np.asarray(X, dtype=float)
        enc_acts = [a]
        for i in range(len(self.encoder_W)):
            a = a @ self.encoder_W[i] + self.encoder_b[i]
            if i < len(self.encoder_W) - 1:
                a = np.maximum(a, 0.0)
            enc_acts.append(a)
        h = enc_acts[-1]
        mu = h @ self.mu_W + self.mu_b
        logvar = h @ self.logvar_W + self.logvar_b
        return mu, logvar, enc_acts

    def _reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*std.shape)
        return mu + eps * std

    def _decode(self, z):
        a = np.asarray(z, dtype=float)
        dec_acts = [a]
        for i in range(len(self.decoder_W) - 1):
            a = np.maximum(a @ self.decoder_W[i] + self.decoder_b[i], 0.0)
            dec_acts.append(a)
        out = dec_acts[-1] @ self.decoder_W[-1] + self.decoder_b[-1]
        dec_acts.append(out)
        return out, dec_acts

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.001, beta: float = 1.0, **kw):
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        beta = float(kw.get("beta", beta))
        n = len(X)
        for _ in range(epochs):
            mu, logvar, enc_acts = self._encode(X)
            z = self._reparameterize(mu, logvar)
            recon, dec_acts = self._decode(z)
            recon_loss = float(np.mean((recon - X) ** 2))
            kl = float(-0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar)))
            loss = recon_loss + beta * kl
            # decoder backprop
            d = 2 * (recon - X) / n
            dec_grads_w = [None] * len(self.decoder_W)
            dec_grads_b = [None] * len(self.decoder_b)
            for i in range(len(self.decoder_W) - 1, -1, -1):
                a_prev = dec_acts[i]
                dec_grads_w[i] = a_prev.T @ d
                dec_grads_b[i] = d.sum(axis=0)
                if i > 0:
                    d = (d @ self.decoder_W[i].T) * (a_prev > 0)
            # latent head grads
            d_mu = mu / n
            d_logvar = (np.exp(logvar) - 1) / (2 * n)
            # grad into last encoder hidden layer
            h_last = enc_acts[-1]
            d_enc = d_mu @ self.mu_W.T + d_logvar @ self.logvar_W.T
            enc_grads_w = [None] * len(self.encoder_W)
            enc_grads_b = [None] * len(self.encoder_b)
            for i in range(len(self.encoder_W) - 1, -1, -1):
                a_prev = enc_acts[i]
                if i == len(self.encoder_W) - 1:
                    h_last = enc_acts[-1]
                    self.mu_W -= lr * (h_last.T @ d_mu)
                    self.mu_b -= lr * d_mu.sum(axis=0)
                    self.logvar_W -= lr * (h_last.T @ d_logvar)
                    self.logvar_b -= lr * d_logvar.sum(axis=0)
                enc_grads_w[i] = a_prev.T @ d_enc
                enc_grads_b[i] = d_enc.sum(axis=0)
                if i > 0:
                    d_enc = (d_enc @ self.encoder_W[i].T) * (a_prev > 0)
            for i in range(len(self.decoder_W)):
                self.decoder_W[i] -= lr * dec_grads_w[i]
                self.decoder_b[i] -= lr * dec_grads_b[i]
            for i in range(len(self.encoder_W)):
                self.encoder_W[i] -= lr * enc_grads_w[i]
                self.encoder_b[i] -= lr * enc_grads_b[i]
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["reconstruction_loss"] = recon_loss
        self.metrics["kl_divergence"] = kl
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def encode(self, X):
        mu, logvar, _ = self._encode(X)
        return mu, logvar

    def decode(self, z):
        return self._decode(np.asarray(z, dtype=float))[0]

    def reconstruct(self, X):
        mu, _, _ = self._encode(X)
        return self._decode(mu)[0]

    def generate(self, n: int = 5, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.latent_dim)).astype(float)
        return self._decode(z)[0]

    def predict(self, X):
        return self.reconstruct(X)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder_W": [w.tolist() for w in self.encoder_W],
            "encoder_b": [b.tolist() for b in self.encoder_b],
            "mu_W": self.mu_W.tolist(),
            "mu_b": self.mu_b.tolist(),
            "logvar_W": self.logvar_W.tolist(),
            "logvar_b": self.logvar_b.tolist(),
            "decoder_W": [w.tolist() for w in self.decoder_W],
            "decoder_b": [b.tolist() for b in self.decoder_b],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.encoder_W = [np.asarray(w, dtype=float) for w in state["encoder_W"]]
        self.encoder_b = [np.asarray(b, dtype=float) for b in state["encoder_b"]]
        self.mu_W = np.asarray(state["mu_W"], dtype=float)
        self.mu_b = np.asarray(state["mu_b"], dtype=float)
        self.logvar_W = np.asarray(state["logvar_W"], dtype=float)
        self.logvar_b = np.asarray(state["logvar_b"], dtype=float)
        self.decoder_W = [np.asarray(w, dtype=float) for w in state["decoder_W"]]
        self.decoder_b = [np.asarray(b, dtype=float) for b in state["decoder_b"]]
        self.trained = True
