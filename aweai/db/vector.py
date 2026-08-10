from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class FlatIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self._vectors.shape[0] > 0:
            self._vectors = np.vstack([self._vectors, vectors])
        else:
            self._vectors = vectors.copy()

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._vectors.shape[0] == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        sims = self._vectors @ query.T
        sims = sims.flatten()
        k = min(k, len(sims))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return sims[idx], idx


class IVFIndex:
    def __init__(self, dim: int, nlist: int = 100) -> None:
        self.dim = dim
        self.nlist = nlist
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._centroids: np.ndarray = np.zeros((nlist, dim), dtype=np.float32)
        self._inverted_lists: List[List[int]] = [[] for _ in range(nlist)]
        self._trained = False

    def _train(self, vectors: np.ndarray) -> None:
        n = len(vectors)
        idx = np.random.choice(n, self.nlist, replace=False)
        centroids = vectors[idx].copy()
        for _ in range(20):
            dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for j in range(self.nlist):
                mask = labels == j
                if np.any(mask):
                    centroids[j] = vectors[mask].mean(axis=0)
        self._centroids = centroids.astype(np.float32)
        self._inverted_lists = [[] for _ in range(self.nlist)]
        for i, label in enumerate(labels):
            self._inverted_lists[label].append(i)
        self._trained = True

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, vectors]) if n > 0 else vectors
        new_ids = np.arange(n, n + len(vectors), dtype=np.int64)
        self._ids = np.concatenate([self._ids, new_ids]) if n > 0 else new_ids
        if not self._trained and len(self._vectors) >= self.nlist:
            self._train(self._vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained or self._vectors.shape[0] == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        sims_to_centroids = self._centroids @ query.T
        sims_to_centroids = sims_to_centroids.flatten()
        nprobe = min(self.nlist, max(1, int(self.nlist * 0.1)))
        top_centroids = np.argpartition(-sims_to_centroids, kth=nprobe - 1)[:nprobe]
        candidates = []
        for c in top_centroids:
            candidates.extend(self._inverted_lists[c])
        if not candidates:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        cand_vecs = self._vectors[candidates]
        sims = cand_vecs @ query.T
        sims = sims.flatten()
        k = min(k, len(sims))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return sims[idx], self._ids[candidates][idx]


class HNSWIndex:
    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200) -> None:
        self.dim = dim
        self.M = M
        self.M_max = M * 2
        self.ef_construction = ef_construction
        self._vectors: Dict[int, np.ndarray] = {}
        self._graphs: List[Dict[int, set]] = []
        self._entry_point: int = 0
        self._max_layer: int = -1
        self._next_id: int = 0

    def _random_level(self) -> int:
        level = 0
        while np.random.random() < 0.5 and level < 16:
            level += 1
        return level

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def _search_layer(self, q: np.ndarray, entry_points: List[int], ef: int, layer: int) -> List[Tuple[float, int]]:
        visited = set(entry_points)
        candidates = [(self._distance(q, self._vectors[ep]), ep) for ep in entry_points]
        candidates.sort(key=lambda x: x[0])
        results = list(candidates)
        while candidates:
            dist, current = candidates[0]
            if dist > results[-1][0]:
                break
            neighbors = self._graphs[layer].get(current, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = self._distance(q, self._vectors[neighbor])
                    if d < results[-1][0] or len(results) < ef:
                        candidates.append((d, neighbor))
                        results.append((d, neighbor))
                        results.sort(key=lambda x: x[0])
                        if len(results) > ef:
                            results = results[:ef]
            candidates.pop(0)
        return results[:ef]

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        for vec in vectors:
            node_id = self._next_id
            self._next_id += 1
            self._vectors[node_id] = vec
            level = self._random_level()
            if self._max_layer < level:
                for l in range(self._max_layer + 1, level + 1):
                    self._graphs.append({})
                self._max_layer = level
            ep = [self._entry_point]
            for l in range(self._max_layer, level, -1):
                res = self._search_layer(vec, ep, 1, l)
                ep = [res[0][1]] if res else ep
            for l in range(min(level, self._max_layer), -1, -1):
                res = self._search_layer(vec, ep, self.ef_construction, l)
                neighbors = [n for _, n in res[:self.M]]
                if len(neighbors) > self.M:
                    neighbors = neighbors[:self.M]
                self._graphs[l][node_id] = set(neighbors)
                for neighbor in neighbors:
                    self._graphs[l][neighbor] = self._graphs[l].get(neighbor, set())
                    self._graphs[l][neighbor].add(node_id)
                    if len(self._graphs[l][neighbor]) > self.M_max:
                        self._graphs[l][neighbor] = set(list(self._graphs[l][neighbor])[:self.M_max])
                ep = neighbors
            if level > 0 or self._next_id == 1:
                self._entry_point = node_id

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._next_id == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        ep = [self._entry_point]
        for l in range(self._max_layer, 0, -1):
            res = self._search_layer(query, ep, 1, l)
            ep = [res[0][1]] if res else ep
        res = self._search_layer(query, ep, max(k, 10), 0)
        sims = np.array([1.0 - d for d, _ in res[:k]], dtype=np.float32)
        ids = np.array([n for _, n in res[:k]], dtype=np.int64)
        return sims, ids


class VectorDB:
    def __init__(self, dim: int = 128, path: Optional[str] = None, index_type: str = "flat") -> None:
        self.dim = dim
        self.path = Path(path) if path else None
        self.index_type = index_type
        if index_type == "hnsw":
            self._index = HNSWIndex(dim)
        elif index_type == "ivf":
            self._index = IVFIndex(dim)
        else:
            self._index = FlatIndex(dim)
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                vectors = np.array(data.get("vectors", []), dtype=np.float32)
                if vectors.size > 0:
                    self._index.add(vectors)
                self._metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
            except Exception:
                pass

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors_list()
        data = {"vectors": vectors.tolist() if vectors.size > 0 else [], "metadata": self._metadata}
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _vectors_list(self) -> np.ndarray:
        if isinstance(self._index, FlatIndex):
            return self._index._vectors
        if isinstance(self._index, IVFIndex):
            return self._index._vectors
        if isinstance(self._index, HNSWIndex):
            if not self._index._vectors:
                return np.zeros((0, self.dim), dtype=np.float32)
            return np.stack([v for v in self._index._vectors.values()])
        return np.zeros((0, self.dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._index.add(vectors)
        ids = list(range(len(self._metadata), len(self._metadata) + len(vectors)))
        for i, mid in enumerate(ids):
            self._metadata[mid] = metadata[i] if metadata and i < len(metadata) else {}
        return ids

    def search(self, query: np.ndarray, k: int = 10, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        sims, idxs = self._index.search(query, k * 2 if filter_fn else k)
        results = []
        for sim, idx in zip(sims, idxs):
            if idx in self._metadata:
                meta = self._metadata[idx]
                if filter_fn and not filter_fn(meta):
                    continue
                results.append({"id": int(idx), "score": float(sim), "metadata": meta})
            if len(results) >= k:
                break
        return results

    def hybrid_search(self, query: np.ndarray, k: int, query_text: str = "", alpha: float = 0.5) -> List[Dict[str, Any]]:
        vec_results = self.search(query, k * 2)
        scored = []
        for r in vec_results:
            text_score = hash(query_text) % 1000 / 1000.0 if query_text else 0.0
            score = alpha * r["score"] + (1 - alpha) * text_score
            scored.append({**r, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def stats(self) -> Dict[str, Any]:
        return {"index_type": self.index_type, "vectors": len(self._metadata), "dim": self.dim}
