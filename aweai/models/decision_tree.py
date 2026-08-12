# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Decision tree (CART) from scratch.

Light-weight, few-resource, yet powerful: a single tree handles both
classification (Gini impurity) and regression (MSE) on a small footprint
with no external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import softmax


def _gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))


def _mse(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


class DecisionTree(BaseModel):
    """CART decision tree (classification or regression via ``criterion``)."""

    model_type = "decision_tree"
    is_classifier = True

    def __init__(self, input_dim: int = 2, max_depth: int = 10,
                 min_samples_split: int = 2, criterion: str = "gini",
                 min_impurity_decrease: float = 0.0, seed: int = 0, **params):
        super().__init__(input_dim=input_dim, max_depth=max_depth,
                         min_samples_split=min_samples_split, criterion=criterion,
                         min_impurity_decrease=min_impurity_decrease, seed=seed, **params)
        self.criterion = criterion
        self.regression = criterion == "mse"
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self._seed = int(seed)
        self.tree: Dict[str, Any] = {}
        self.n_classes_: int = 0
        self.classes_: np.ndarray = np.array([])

    # ----------------------------------------------------------- training
    def _best_split(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray):
        best_gain = -1.0
        best = None
        n = len(idx)
        if n < 2:
            return None
        rng = np.random.default_rng(self._seed)
        for feat in rng.permutation(X.shape[1]):
            vals = np.unique(X[idx, feat])
            for thr in vals:
                left = idx[X[idx, feat] <= thr]
                right = idx[X[idx, feat] > thr]
                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue
                if self.regression:
                    gain = _mse(y) - (len(left) / n) * _mse(y[left]) - (len(right) / n) * _mse(y[right])
                else:
                    gain = _gini(y) - (len(left) / n) * _gini(y[left]) - (len(right) / n) * _gini(y[right])
                if gain > best_gain:
                    best_gain = gain
                    best = (feat, float(thr), left, right)
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict[str, Any]:
        node: Dict[str, Any] = {}
        if self.regression:
            node["value"] = float(y.mean())
        else:
            counts = np.bincount(y.astype(int), minlength=self.n_classes_)
            node["value"] = int(np.argmax(counts))
            node["dist"] = (counts / counts.sum()).tolist()
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return node
        split = self._best_split(X, y, np.arange(len(y)))
        if split is None or self._gain(split, X, y) <= self.min_impurity_decrease + 1e-9:
            return node
        feat, thr, left, right = split
        node["feat"] = feat
        node["thr"] = thr
        node["left"] = self._build(X[left], y[left], depth + 1)
        node["right"] = self._build(X[right], y[right], depth + 1)
        # drop dist/value leaf-only marker when split to save space
        node.pop("value", None)
        node.pop("dist", None)
        return node

    def _gain(self, split, X, y) -> float:
        feat, thr, left, right = split
        n = len(y)
        if self.regression:
            return _mse(y) - (len(left) / n) * _mse(y[left]) - (len(right) / n) * _mse(y[right])
        return _gini(y) - (len(left) / n) * _gini(y[left]) - (len(right) / n) * _gini(y[right])

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is None:
            y = np.zeros(len(X), dtype=int)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = max(int(self.classes_.max()) + 1 if self.classes_.size else 1, 1)
        if self.regression:
            y = y.astype(float).reshape(-1)
        else:
            y = y.astype(int).reshape(-1)
        # if labels are not 0..n-1 contiguous ints, remap
        if not self.regression and (y.max() >= self.n_classes_ or y.min() < 0):
            remap = {c: i for i, c in enumerate(np.sort(self.classes_))}
            y = np.array([remap[v] for v in y])
            self._remap = remap
            self._classes_ordered = np.sort(self.classes_)
        else:
            self._remap = None
            self._classes_ordered = self.classes_
        self.tree = self._build(X, y, 0)
        self.trained = True
        if self.regression:
            pred = self.predict(X)
            self.metrics["mse"] = float(np.mean((pred - y) ** 2))
        else:
            pred = self.predict(X)
            self.metrics["accuracy"] = float(np.mean(pred == y))
        return self

    # ----------------------------------------------------------- inference
    def _predict_one(self, x: np.ndarray) -> Any:
        node = self.tree
        while "left" in node:
            if x[node["feat"]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        if self.regression:
            return node["value"]
        if self._remap is not None:
            return self._classes_ordered[node["value"]]
        return node["value"]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.array([self._predict_one(x) for x in X])
        if not self.regression:
            out = out.astype(int)
        return out

    def predict_proba(self, X):
        if self.regression:
            return self.predict(X)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = []
        for x in X:
            node = self.tree
            while "left" in node:
                if x[node["feat"]] <= node["thr"]:
                    node = node["left"]
                else:
                    node = node["right"]
            if "dist" in node:
                out.append(node["dist"])
            else:
                vec = np.zeros(self.n_classes_, dtype=float)
                vec[int(node["value"])] = 1.0
                out.append(vec.tolist())
        return np.array(out)

    def state_dict(self) -> Dict[str, Any]:
        return {"tree": self.tree, "n_classes": self.n_classes_,
                "classes": self.classes_.tolist(),
                "regression": self.regression,
                "remap": (self._remap or {}),
                "classes_ordered": self._classes_ordered.tolist() if self._classes_ordered is not None else []}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.tree = state["tree"]
        self.n_classes_ = int(state.get("n_classes", 2))
        self.classes_ = np.asarray(state.get("classes", []), dtype=int) if not self.regression else self.classes_
        self.regression = bool(state.get("regression", False))
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
        self._remap = state.get("remap") or None
        co = state.get("classes_ordered", [])
        self._classes_ordered = np.asarray(co, dtype=int) if co else np.array([])
        self.trained = True
