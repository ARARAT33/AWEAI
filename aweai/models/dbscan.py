# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""DBSCAN density-based clustering from scratch.

A classic but powerful clustering algorithm that finds arbitrarily-shaped
clusters and marks noise points — implemented with only numpy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


class DBSCAN(BaseModel):
    """Density-based spatial clustering of applications with noise."""

    model_type = "dbscan"
    is_clusterer = True

    def __init__(self, input_dim: int = 2, eps: float = 0.5, min_samples: int = 5,
                 metric: str = "euclidean", seed: int = 0, **params):
        super().__init__(input_dim=input_dim, eps=float(eps),
                         min_samples=int(min_samples), metric=metric, seed=seed, **params)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric
        self._seed = int(seed)
        self.labels_: np.ndarray = np.array([])
        self.components_: np.ndarray = np.array([])
        self._neighbors: Dict[int, np.ndarray] = {}

    def _dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.metric == "manhattan":
            return np.sum(np.abs(a - b), axis=1)
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(X)
        visited = np.zeros(n, dtype=bool)
        labels = np.full(n, -1)
        # precompute neighbor lists within eps for core points
        neighbors: Dict[int, np.ndarray] = {}
        for i in range(n):
            dists = self._dist(X, X[i])
            idx = np.where(dists <= self.eps)[0]
            idx = idx[idx != i]
            neighbors[i] = idx
        self._neighbors = neighbors
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            if len(neighbors[i]) < self.min_samples:
                continue
            # expand cluster
            labels[i] = cluster_id
            seed_set = set(neighbors[i].tolist())
            while seed_set:
                j = seed_set.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id
                if visited[j]:
                    continue
                visited[j] = True
                if len(neighbors[j]) >= self.min_samples:
                    seed_set.update(neighbors[j].tolist())
            cluster_id += 1
        self.labels_ = labels
        self.n_clusters_ = int(max(labels.max() + 1, 0))
        core_idx = np.array([i for i, v in neighbors.items() if len(v) >= self.min_samples])
        self._core_idx = core_idx
        self.components_ = X[core_idx] if len(core_idx) else X[:0]
        n_noise = int(np.sum(labels == -1))
        self.trained = True
        self.metrics["n_clusters"] = self.n_clusters_
        self.metrics["n_noise"] = n_noise
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = []
        for x in X:
            dists = self._dist(self.components_, x) if len(self.components_) else np.array([np.inf])
            if self.n_clusters_ == 0 or len(self.components_) == 0:
                out.append(-1)
                continue
            nn = int(np.argmin(dists))
            if dists[nn] > self.eps:
                out.append(-1)
            else:
                out.append(int(self.labels_[self._core_idx[nn]]))
        return np.array(out)

    def fit_predict(self, X, y=None, **kw):
        self.fit(X, y=y, **kw)
        return self.labels_

    def state_dict(self) -> Dict[str, Any]:
        return {"labels": self.labels_.tolist(), "eps": self.eps,
                "min_samples": self.min_samples, "n_clusters": getattr(self, "n_clusters_", 0),
                "components": self.components_.tolist(),
                "core_idx": getattr(self, "_core_idx", np.array([])).tolist()}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.labels_ = np.asarray(state["labels"], dtype=int)
        self.eps = float(state.get("eps", self.eps))
        self.min_samples = int(state.get("min_samples", self.min_samples))
        self.n_clusters_ = int(state.get("n_clusters", 0))
        self.components_ = np.asarray(state.get("components", []), dtype=float)
        self._core_idx = np.asarray(state.get("core_idx", []), dtype=int)
        self.trained = True
