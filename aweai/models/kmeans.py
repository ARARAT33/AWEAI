"""K-Means clustering (Lloyd's algorithm, numpy)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from aweai.models.base import BaseModel


class KMeans(BaseModel):
    model_type = "kmeans"
    is_clusterer = True

    def __init__(self, k: int = 3, max_iter: int = 100, seed: int = 0, **params):
        super().__init__(k=k, max_iter=max_iter, **params)
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.centroids: np.ndarray = np.zeros((self.k, 1))
        self.labels_: np.ndarray = np.zeros(0, dtype=int)

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(kw.get("seed", self.params.get("seed", 0)))
        idx = rng.choice(len(X), size=min(self.k, len(X)), replace=False)
        self.centroids = X[idx].copy()
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_c = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j] for j in range(self.k)])
            if np.allclose(new_c, self.centroids):
                break
            self.centroids = new_c
        self.labels_ = labels
        self.trained = True
        self.metrics["inertia"] = float(np.sum(np.min(np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2), axis=1) ** 2))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def state_dict(self):
        return {"centroids": self.centroids.tolist()}

    def load_state(self, state):
        self.centroids = np.asarray(state["centroids"], dtype=float)
        self.trained = True
