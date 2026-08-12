# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Naive Bayes from scratch (Gaussian + Bernoulli).

Tiny, fast probabilistic classifiers with near-zero memory overhead — ideal
for low-resource environments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = a.max()
    if not np.isfinite(m):
        return float(a.max())
    return float(m + np.log(np.sum(np.exp(a - m))))


class NaiveBayes(BaseModel):
    """Gaussian (default) / Bernoulli Naive Bayes."""

    model_type = "naive_bayes"
    is_classifier = True

    def __init__(self, input_dim: int = 2, distribution: str = "gaussian",
                 var_smoothing: float = 1e-9, seed: int = 0, **params):
        super().__init__(input_dim=input_dim, distribution=distribution,
                         var_smoothing=var_smoothing, seed=seed, **params)
        self.distribution = distribution
        self.var_smoothing = float(var_smoothing)
        self._seed = int(seed)
        self.classes_: np.ndarray = np.array([])
        self._mean: np.ndarray = np.array([])
        self._var: np.ndarray = np.array([])
        self._priors: np.ndarray = np.array([])

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if self.distribution == "bernoulli":
            # binarize on a threshold of the global mean
            X = (X > X.mean(axis=0)).astype(float)
        self._mean = np.zeros((n_classes, X.shape[1]))
        self._var = np.zeros((n_classes, X.shape[1]))
        self._bern = np.zeros((n_classes, X.shape[1]))
        self._priors = np.zeros(n_classes)
        for ci, c in enumerate(self.classes_):
            Xc = X[y == c]
            self._priors[ci] = len(Xc) / len(X)
            if self.distribution == "bernoulli":
                self._bern[ci] = Xc.mean(axis=0) + 1e-9
            else:
                self._mean[ci] = Xc.mean(axis=0)
                self._var[ci] = Xc.var(axis=0) + self.var_smoothing
        self.trained = True
        self.metrics["accuracy"] = float(np.mean(self.predict(X) == y))
        return self

    def _joint(self, x: np.ndarray) -> np.ndarray:
        if self.distribution == "bernoulli":
            eps = 1e-9
            x = (x > 0).astype(float)
            p = self._bern
            return self._priors + np.sum(x * np.log(p + eps) + (1 - x) * np.log(1 - p + eps), axis=1)
        # gaussian
        mean = self._mean
        var = self._var
        # log pdf per class per feature
        logp = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean) ** 2 / var, axis=1)
        return np.log(self._priors + 1e-300) + logp

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = []
        for x in X:
            scores = self._joint(x)
            out.append(int(self.classes_[int(np.argmax(scores))]))
        return np.array(out)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rows = []
        for x in X:
            scores = self._joint(x)
            m = _logsumexp(scores)
            probs = np.exp(scores - m)
            rows.append(probs)
        return np.array(rows)

    def state_dict(self) -> Dict[str, Any]:
        return {"classes": self.classes_.tolist(), "mean": self._mean.tolist(),
                "var": self._var.tolist(), "bern": self._bern.tolist(),
                "priors": self._priors.tolist(), "distribution": self.distribution}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.classes_ = np.asarray(state["classes"], dtype=int)
        self._mean = np.asarray(state["mean"], dtype=float)
        self._var = np.asarray(state["var"], dtype=float)
        self._bern = np.asarray(state.get("bern", []), dtype=float)
        self._priors = np.asarray(state["priors"], dtype=float)
        self.distribution = state.get("distribution", "gaussian")
        self.trained = True
