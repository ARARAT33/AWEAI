# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""K-Nearest Neighbors from scratch.

Lazy learner: training is essentially free (just stores the data), so the
model footprint is predictable and tiny at train time. Prediction cost scales
with dataset size — excellent for small/low-resource datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


class KNN(BaseModel):
    """K-Nearest Neighbors (classification or regression)."""

    model_type = "knn"
    is_classifier = True

    def __init__(self, input_dim: int = 2, k: int = 5, weights: str = "uniform",
                 metric: str = "euclidean", seed: int = 0, **params):
        super().__init__(input_dim=input_dim, k=k, weights=weights,
                         metric=metric, seed=seed, **params)
        self.k = int(k)
        self.weights = weights
        self.metric = metric
        self._seed = int(seed)
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])
        self.regression = False

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is None:
            y = np.zeros(len(X), dtype=int)
        y = np.asarray(y)
        # regression if y is continuous (float with many unique values)
        if y.dtype.kind == "f" and len(np.unique(y)) > 10:
            self.regression = True
            self.is_classifier = False
            self.is_regressor = True
            y = y.astype(float)
        else:
            self.regression = False
            self.is_classifier = True
            self.is_regressor = False
            y = y.astype(int)
        self.X_train = X
        self.y_train = y
        self.classes_ = (np.unique(y) if not self.regression else np.array([]))
        self.trained = True
        if not self.regression:
            self.metrics["accuracy"] = float(np.mean(self.predict(X) == y))
        else:
            self.metrics["mse"] = float(np.mean((self.predict(X) - y) ** 2))
        return self

    def _dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.metric == "manhattan":
            return np.sum(np.abs(a - b), axis=1)
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def _vote(self, neighbor_labels: np.ndarray) -> Any:
        if self.regression:
            if self.weights == "distance":
                # weighted by inverse distance handled by caller for distance-weighted
                return float(np.mean(neighbor_labels))
            return float(np.mean(neighbor_labels))
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        if self.weights == "distance":
            return vals[np.argmax(counts)]
        return vals[np.argmax(counts)]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = []
        k = min(self.k, len(self.X_train))
        for x in X:
            dists = self._dist(self.X_train, x)
            nn = np.argpartition(dists, k)[:k]
            dists_nn = dists[nn]
            labels = self.y_train[nn]
            if self.weights == "distance":
                w = 1.0 / (dists_nn + 1e-9)
                if self.regression:
                    val = float(np.average(labels, weights=w))
                else:
                    vals, counts = np.unique(labels, return_counts=True)
                    wsum = np.array([w[labels == v].sum() for v in vals])
                    val = int(vals[np.argmax(wsum)])
            else:
                val = self._vote(labels)
            out.append(val)
        return np.array(out)

    def predict_proba(self, X):
        if self.regression:
            return self.predict(X)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rows = []
        k = min(self.k, len(self.X_train))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for x in X:
            dists = self._dist(self.X_train, x)
            nn = np.argpartition(dists, k)[:k]
            labels = self.y_train[nn]
            if self.weights == "distance":
                w = 1.0 / (dists[nn] + 1e-9)
            else:
                w = np.ones(len(labels))
            proba = np.zeros(len(self.classes_), dtype=float)
            for lbl, wgt in zip(labels, w):
                if int(lbl) in class_to_idx:
                    proba[class_to_idx[int(lbl)]] += wgt
            proba = proba / proba.sum() if proba.sum() else proba
            rows.append(proba.tolist())
        return np.array(rows)

    def state_dict(self) -> Dict[str, Any]:
        return {"X_train": self.X_train.tolist(), "y_train": self.y_train.tolist(),
                "classes": self.classes_.tolist(), "regression": self.regression,
                "k": self.k, "weights": self.weights, "metric": self.metric}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.X_train = np.asarray(state["X_train"], dtype=float)
        self.y_train = np.asarray(state["y_train"])
        self.classes_ = np.asarray(state.get("classes", []), dtype=int)
        self.k = int(state.get("k", self.k))
        self.weights = state.get("weights", self.weights)
        self.metric = state.get("metric", self.metric)
        self.regression = bool(state.get("regression", False))
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
        self.trained = True
