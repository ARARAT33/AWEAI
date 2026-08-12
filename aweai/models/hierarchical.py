# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Agglomerative hierarchical clustering from scratch.

Bottom-up clustering with single / complete / average / Ward linkage, built
purely on numpy. Cutting the dendrogram at ``n_clusters`` yields labels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from aweai.models.base import BaseModel


class Hierarchical(BaseModel):
    """Agglomerative (hierarchical) clustering."""

    model_type = "hierarchical"
    is_clusterer = True

    def __init__(self, input_dim: int = 2, n_clusters: int = 3,
                 linkage: str = "ward", metric: str = "euclidean",
                 seed: int = 0, **params):
        super().__init__(input_dim=input_dim, n_clusters=int(n_clusters),
                         linkage=linkage, metric=metric, seed=seed, **params)
        self.n_clusters = int(n_clusters)
        self.linkage = linkage
        self.metric = metric
        self._seed = int(seed)
        self.labels_: np.ndarray = np.array([])
        self.linkage_matrix_: List[List[float]] = []
        self.children_: Dict[int, Any] = {}
        self.centroids_: np.ndarray = np.array([])
        self.n_clusters_: int = 0

    def _dist_mat(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - X[None, :, :]
        if self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=2)
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(X)
        D = self._dist_mat(X)
        np.fill_diagonal(D, np.inf)
        active: List[Dict[str, Any]] = [{"id": i, "members": {i}, "w": 1.0} for i in range(n)]
        linkage_matrix: List[List[float]] = []
        children: Dict[int, Any] = {}
        nid = n
        while len(active) > 1:
            mind = np.inf
            ai = aj = -1
            for ci in range(len(active)):
                for cj in range(ci + 1, len(active)):
                    d = self._pair_dist(D, active[ci]["members"], active[cj]["members"],
                                        active[ci]["w"], active[cj]["w"])
                    if d < mind:
                        mind = d
                        ai, aj = ci, cj
            if ai < 0:
                break
            na, nb = active[ai]["id"], active[aj]["id"]
            size = active[ai]["w"] + active[aj]["w"]
            linkage_matrix.append([float(na), float(nb), round(float(mind), 6), float(size)])
            children[nid] = (na, nb)
            merged = active[ai]["members"] | active[aj]["members"]
            active = [c for k, c in enumerate(active) if k not in (ai, aj)]
            active.append({"id": nid, "members": merged, "w": size})
            nid += 1
        self.linkage_matrix_ = linkage_matrix
        self.children_ = children
        self.labels_ = self._cut(n)
        self.n_clusters_ = int(max(self.labels_.max() + 1, 1)) if len(self.labels_) else 0
        if self.n_clusters_ > 0:
            self.centroids_ = np.array([X[self.labels_ == c].mean(axis=0)
                                        for c in range(self.n_clusters_)])
        else:
            self.centroids_ = np.zeros((0, X.shape[1]))
        self.trained = True
        self.metrics["n_clusters"] = self.n_clusters_
        return self

    def _pair_dist(self, D, c1: Set[int], c2: Set[int], w1: float, w2: float) -> float:
        ids1 = list(c1)
        ids2 = list(c2)
        sub = D[np.ix_(ids1, ids2)]
        if self.linkage == "single":
            return float(sub.min())
        if self.linkage == "complete":
            return float(sub.max())
        if self.linkage == "average":
            return float(sub.mean())
        # ward
        return float(w1 * w2 / (w1 + w2) * sub.mean())

    def _cut(self, n: int) -> np.ndarray:
        target = max(1, min(self.n_clusters, n))
        # the final tree root is the last merge id = n + len(linkage) - 1
        root = n + len(self.linkage_matrix_) - 1
        groups = [root]
        while len(groups) < target:
            nxt = None
            for g in groups:
                if g in self.children_:
                    nxt = g
                    break
            if nxt is None:
                break
            groups.remove(nxt)
            groups.extend(self.children_[nxt])
        labels = np.full(n, 0, dtype=int)
        for cid, node in enumerate(groups):
            for leaf in self._leaf_set(node):
                labels[leaf] = cid
        return labels

    def _leaf_set(self, node: int) -> List[int]:
        out = []
        stack = [node]
        while stack:
            nd = stack.pop()
            if nd in self.children_:
                stack.append(self.children_[nd][0])
                stack.append(self.children_[nd][1])
            else:
                out.append(nd)
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.centroids_.shape[0] == 0:
            return np.full(len(X), -1)
        diff = X[:, None, :] - self.centroids_[None, :, :]
        if self.metric == "manhattan":
            d = np.sum(np.abs(diff), axis=2)
        else:
            d = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.argmin(d, axis=1)

    def fit_predict(self, X, y=None, **kw):
        self.fit(X, y=y, **kw)
        return self.labels_

    def state_dict(self) -> Dict[str, Any]:
        return {"labels": self.labels_.tolist(),
                "linkage_matrix": self.linkage_matrix_,
                "children": self.children_,
                "centroids": self.centroids_.tolist(),
                "n_clusters": self.n_clusters, "linkage": self.linkage}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.labels_ = np.asarray(state["labels"], dtype=int)
        self.linkage_matrix_ = state.get("linkage_matrix", [])
        self.children_ = {int(k): v for k, v in state.get("children", {}).items()}
        self.centroids_ = np.asarray(state.get("centroids", []), dtype=float)
        self.n_clusters = int(state.get("n_clusters", self.n_clusters))
        self.n_clusters_ = int(max(self.labels_.max() + 1, 1)) if len(self.labels_) else 0
        self.trained = True
