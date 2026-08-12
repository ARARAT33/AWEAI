# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Random forest from scratch.

An ensemble of decision trees built on bootstrap samples with per-tree
feature subsampling. Powerful yet frugal: with small ``n_estimators`` and
``max_depth`` it classifies well on CPU with a tiny footprint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.models.decision_tree import DecisionTree


class RandomForest(BaseModel):
    """Random forest: bagged + feature-subsampled decision trees."""

    model_type = "random_forest"
    is_classifier = True

    def __init__(self, input_dim: int = 2, n_estimators: int = 10,
                 max_depth: int = 10, min_samples_split: int = 2,
                 criterion: str = "gini", max_features: str = "sqrt",
                 n_jobs: int = 1, seed: int = 0, **params):
        super().__init__(input_dim=input_dim, n_estimators=n_estimators,
                         max_depth=max_depth, min_samples_split=min_samples_split,
                         criterion=criterion, max_features=max_features,
                         n_jobs=n_jobs, seed=seed, **params)
        self.regression = criterion == "mse"
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.criterion = criterion
        self.max_features = max_features
        self.n_jobs = int(n_jobs)
        self._seed = int(seed)
        self.trees: List[DecisionTree] = []
        self.classes_: np.ndarray = np.array([])

    def _n_feat(self, n_total: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(round(np.sqrt(n_total))))
        if self.max_features == "log2":
            return max(1, int(round(np.log2(n_total))))
        if isinstance(self.max_features, int):
            return max(1, min(int(self.max_features), n_total))
        return max(1, n_total // 3 if n_total > 3 else 1)

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is None:
            y = np.zeros(len(X), dtype=int)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self._seed)
        feat_idx = np.arange(X.shape[1])
        self.trees = []
        for i in range(self.n_estimators):
            n_feat = self._n_feat(X.shape[1])
            sel = rng.choice(feat_idx, size=n_feat, replace=False)
            sel = np.sort(sel)
            sample_idx = rng.integers(0, len(X), size=len(X))  # bootstrap
            tree = DecisionTree(input_dim=X.shape[1], max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                criterion=self.criterion, seed=self._seed + i)
            Xb = X[sample_idx][:, sel] if len(sel) < X.shape[1] else X[sample_idx]
            tree._feat_map = sel
            tree.fit(Xb, y[sample_idx])
            self.trees.append(tree)
        self.regression = self.trees[0].regression if self.trees else False
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
        self.trained = True
        if not self.regression and y is not None:
            self.metrics["accuracy"] = float(np.mean(self.predict(X) == y))
        elif self.regression:
            self.metrics["mse"] = float(np.mean((self.predict(X) - y.astype(float)) ** 2))
        return self

    def _predict_one(self, x: np.ndarray, tree: DecisionTree) -> Any:
        node = tree.tree
        fmap = np.asarray(tree._feat_map, dtype=int)
        while "left" in node:
            if x[fmap[node["feat"]]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        if tree.regression:
            return node["value"]
        val = node["value"] if "value" in node else int(np.argmax(node.get("dist", [0])))
        remap = getattr(tree, "_remap", None)
        if remap:
            return tree._classes_ordered[val]
        return val

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.regression:
            preds = np.array([[self._predict_one(x, t) for t in self.trees] for x in X])
            return preds.mean(axis=1)
        votes = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for t in self.trees:
            for i, x in enumerate(X):
                v = self._predict_one(x, t)
                if v in class_to_idx:
                    votes[i, class_to_idx[v]] += 1.0
        return np.array([self.classes_[int(np.argmax(r))] for r in votes])

    def predict_proba(self, X):
        if self.regression:
            return self.predict(X)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        votes = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for t in self.trees:
            for i, x in enumerate(X):
                v = self._predict_one(x, t)
                if v in class_to_idx:
                    votes[i, class_to_idx[v]] += 1.0
        return votes / max(len(self.trees), 1)

    def state_dict(self) -> Dict[str, Any]:
        trees_state = []
        for tree in self.trees:
            trees_state.append({
                "feat_map": getattr(tree, "_feat_map", np.arange(tree.input_dim)).tolist(),
                "tree": tree.tree,
                "n_classes": tree.n_classes_,
                "regression": tree.regression,
                "remap": getattr(tree, "_remap", None),
                "classes_ordered": (getattr(tree, "_classes_ordered", np.array([])).tolist()
                                    if hasattr(getattr(tree, "_classes_ordered", None), "tolist")
                                    else list(getattr(tree, "_classes_ordered", []) or [])),
                "input_dim": tree.input_dim,
                "max_depth": tree.max_depth,
                "min_samples_split": tree.min_samples_split,
                "criterion": tree.criterion,
                "seed": tree._seed,
            })
        return {"trees": trees_state, "n_classes": int(len(self.classes_)),
                "classes": self.classes_.tolist(), "regression": self.regression}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.trees = []
        for ts in state["trees"]:
            tree = DecisionTree(input_dim=ts.get("input_dim", 2), max_depth=ts.get("max_depth", 10),
                                min_samples_split=ts.get("min_samples_split", 2),
                                criterion=ts.get("criterion", "gini"), seed=ts.get("seed", 0))
            tree._feat_map = np.asarray(ts.get("feat_map", []), dtype=int)
            tree.tree = ts["tree"]
            tree.n_classes_ = int(ts.get("n_classes", 2))
            tree.regression = bool(ts.get("regression", False))
            tree._remap = ts.get("remap")
            co = ts.get("classes_ordered", [])
            tree._classes_ordered = np.asarray(co, dtype=int) if co else np.array([])
            tree.trained = True
            self.trees.append(tree)
        self.classes_ = np.asarray(state.get("classes", []), dtype=int)
        self.regression = bool(state.get("regression", False))
        if self.regression:
            self.is_classifier = False
            self.is_regressor = True
        self.trained = True
