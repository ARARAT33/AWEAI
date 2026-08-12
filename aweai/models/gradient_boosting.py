# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Gradient boosting from scratch.

Sequential ensemble of shallow decision trees fit on the (negative) gradient
of the loss. With ``max_depth=1`` (stumps) and a modest ``n_estimators`` it
is a powerful learner that stays tiny in memory and fast on CPU.

* ``objective="regression"`` -> least-squares gradient boosting.
* ``objective="binary"``    -> logistic loss (2 classes).
* ``objective="multiclass"`` -> one-vs-rest logistic boosting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.models.decision_tree import DecisionTree
from aweai.utils import sigmoid


class GradientBoosting(BaseModel):
    """Gradient boosting machine (regression + binary / multiclass classification)."""

    model_type = "gradient_boosting"
    is_classifier = True

    def __init__(self, input_dim: int = 2, n_estimators: int = 50,
                 max_depth: int = 2, lr: float = 0.1, objective: str = "auto",
                 subsample: float = 1.0, seed: int = 0, **params):
        super().__init__(input_dim=input_dim, n_estimators=n_estimators,
                         max_depth=max_depth, lr=lr, objective=objective,
                         subsample=subsample, seed=seed, **params)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.lr = float(lr)
        self.objective = objective
        self.subsample = float(subsample)
        self._seed = int(seed)
        # trees: list of {"tree": ..., "max_depth": ...} for regression/binary
        # trees_by_class: {class_label: [tree_info, ...]} for multiclass
        self.trees: List[Dict[str, Any]] = []
        self.trees_by_class: Dict[int, List[Dict[str, Any]]] = {}
        self.init_: Any = 0.0
        self.classes_: np.ndarray = np.array([])
        self.regression = False

    # ----------------------------------------------------------- inference
    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        """Sum of (lr * tree.predict(X)) over all trees (regression / binary)."""
        score = np.full(len(X), float(self.init_))
        for t in self.trees:
            tree = DecisionTree(input_dim=X.shape[1], max_depth=t["max_depth"],
                                min_samples_split=2, criterion="mse")
            tree.tree = t["tree"]
            score = score + self.lr * tree.predict(X)
        return score

    def _raw_score_multiclass(self, X: np.ndarray) -> np.ndarray:
        raw = np.zeros((len(X), len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            for t in self.trees_by_class[int(c)]:
                tree = DecisionTree(input_dim=X.shape[1], max_depth=t["max_depth"],
                                    min_samples_split=2, criterion="mse")
                tree.tree = t["tree"]
                raw[:, ci] = raw[:, ci] + self.lr * tree.predict(X)
        return raw

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.regression:
            return self._raw_score(X)
        if len(self.classes_) == 2:
            proba = sigmoid(self._raw_score(X))
            return np.where(proba >= 0.5, self.classes_[1], self.classes_[0])
        raw = self._raw_score_multiclass(X)
        return self.classes_[np.argmax(raw, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.regression:
            return self._raw_score(X)
        if len(self.classes_) == 2:
            p = sigmoid(self._raw_score(X)).reshape(-1, 1)
            return np.hstack([1 - p, p])
        from aweai.utils import softmax
        return softmax(self._raw_score_multiclass(X))

    # ----------------------------------------------------------- training
    def _fit_binary(self, X, y_binary):
        rng = np.random.default_rng(self._seed)
        self.init_ = 0.0
        self.trees = []
        for i in range(self.n_estimators):
            score = self._raw_score(X)
            proba = sigmoid(score)
            grad = y_binary - proba  # gradient of logloss
            tree = DecisionTree(input_dim=X.shape[1], max_depth=self.max_depth,
                                min_samples_split=2, criterion="mse", seed=self._seed + i)
            tree.fit(X, grad)
            self.trees.append({"tree": tree.tree, "max_depth": self.max_depth})
        return self

    def _fit_multiclass(self, X, y):
        self.trees_by_class = {}
        for c in self.classes_:
            y_binary = (y == c).astype(float)
            # temporary shared trees list
            self.trees = []
            self._fit_binary(X, y_binary)
            self.trees_by_class[int(c)] = list(self.trees)
        self.trees = []
        return self

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        is_float = y.dtype.kind == "f"
        if self.objective == "auto":
            objective = "regression" if is_float else "multiclass"
        else:
            objective = self.objective
        self.regression = objective == "regression"
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
            target = y.astype(float).reshape(-1)
            self.init_ = float(target.mean())
            rng = np.random.default_rng(self._seed)
            for i in range(self.n_estimators):
                resid = target - self._raw_score(X)
                if self.subsample < 1.0:
                    n_sub = max(1, int(self.subsample * len(X)))
                    idx = rng.choice(len(X), size=n_sub, replace=False)
                    tree = DecisionTree(input_dim=X.shape[1], max_depth=self.max_depth,
                                        min_samples_split=2, criterion="mse", seed=self._seed + i)
                    tree.fit(X[idx], resid[idx])
                else:
                    tree = DecisionTree(input_dim=X.shape[1], max_depth=self.max_depth,
                                        min_samples_split=2, criterion="mse", seed=self._seed + i)
                    tree.fit(X, resid)
                self.trees.append({"tree": tree.tree, "max_depth": self.max_depth})
            self.metrics["mse"] = float(np.mean((self.predict(X) - y.astype(float)) ** 2))
        else:
            self.classes_ = np.unique(y)
            if objective == "binary" or len(self.classes_) == 2:
                y_binary = np.where(y == self.classes_[0], -1.0, 1.0)
                self._fit_binary(X, y_binary)
                self.metrics["accuracy"] = float(np.mean(self.predict(X) == y))
            else:
                self._fit_multiclass(X, y)
                self.metrics["accuracy"] = float(np.mean(self.predict(X) == y))
        self.trained = True
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"trees": self.trees, "trees_by_class": self.trees_by_class, "init": self.init_,
                "classes": self.classes_.tolist(), "regression": self.regression,
                "objective": self.objective}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.trees = state.get("trees", [])
        self.trees_by_class = {int(k): v for k, v in state.get("trees_by_class", {}).items()}
        self.init_ = state.get("init", 0.0)
        self.classes_ = np.asarray(state.get("classes", []), dtype=int)
        self.regression = bool(state.get("regression", False))
        self.objective = state.get("objective", self.objective)
        self.is_classifier = not self.regression
        self.is_regressor = self.regression
        self.trained = True
