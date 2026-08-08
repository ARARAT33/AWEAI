"""Normalization / encoding helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def standardize(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std, {"mean": mean.ravel().tolist(), "std": std.ravel().tolist()}


def minmax(X: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
    X = np.asarray(X, dtype=float)
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    span = mx - mn
    span[span == 0] = 1.0
    out = lo + (X - mn) / span * (hi - lo)
    return out, {"min": mn.ravel().tolist(), "max": mx.ravel().tolist()}


def normalize_numeric(X: np.ndarray, method: str = "standardize") -> Tuple[np.ndarray, Dict[str, float]]:
    method = (method or "standardize").lower()
    if method in ("standardize", "zscore", "z"):
        return standardize(X)
    if method in ("minmax", "scale"):
        return minmax(X)
    raise ValueError(f"Unknown normalization method: {method}")


def one_hot(labels: Sequence[Any]) -> Tuple[np.ndarray, List[Any]]:
    uniq = list(dict.fromkeys(labels))
    mapping = {v: i for i, v in enumerate(uniq)}
    idx = np.array([mapping[v] for v in labels], dtype=np.int64)
    n = len(idx)
    k = len(uniq)
    mat = np.zeros((n, k), dtype=float)
    mat[np.arange(n), idx] = 1.0
    return mat, uniq


def label_encode(labels: Sequence[Any]) -> Tuple[np.ndarray, List[Any]]:
    uniq = list(dict.fromkeys(labels))
    mapping = {v: i for i, v in enumerate(uniq)}
    return np.array([mapping[v] for v in labels], dtype=np.int64), uniq
