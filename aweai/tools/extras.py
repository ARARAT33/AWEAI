"""AWEAI extras — extension toolkit for the model factory.

This package is the growth point for turning AWEAI into a full model
factory: feature engineering, ensembling, quantization, pruning,
distillation, synthetic data, hyperparameter search, model comparison,
deployment helpers, monitoring and visualization utilities land here as
small, dependency-light modules.

Current utilities (all numpy-only):
    - featurize      : basic feature engineering (polynomial, one-hot, bins)
    - make_synthetic : tiny synthetic dataset generators (blobs, XOR, sine)
    - ensemble_vote  : hard/soft voting across fitted model predictions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def featurize(X: Sequence[Sequence[float]], degree: int = 2, with_bias: bool = True) -> np.ndarray:
    """Expand numeric features with polynomial terms.

    For each sample this returns [1 (if bias), x1..xd, x1^2, x1*x2, ...].
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = []
    if with_bias:
        out.append(np.ones(len(arr)))
    for d in range(1, degree + 1):
        # all monomials of degree d over the d features (including repeats)
        cols = arr.shape[1]
        for combo in _monomials(cols, d):
            col = np.ones(len(arr))
            for c in combo:
                col = col * arr[:, c]
            out.append(col)
    return np.column_stack(out) if out else arr


def _monomials(n: int, degree: int):
    if degree == 1:
        for i in range(n):
            yield [i]
        return
    for prev in _monomials(n, degree - 1):
        for i in range(prev[-1], n):
            yield prev + [i]


def make_synthetic(kind: str = "blobs", n: int = 120, seed: int = 0) -> Dict[str, Any]:
    """Generate small synthetic datasets for demos and tests.

    kind: "blobs" (classification), "xor" (classification),
          "sine" (regression), "clusters" (clustering).
    """
    rng = np.random.default_rng(seed)
    if kind == "blobs":
        X = np.vstack([rng.normal([0, 0], 0.6, (n // 2, 2)), rng.normal([3, 3], 0.6, (n - n // 2, 2))])
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        return {"X": X, "y": y, "task": "classification"}
    if kind == "xor":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
        return {"X": X, "y": y, "task": "classification"}
    if kind == "sine":
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.1, n)
        return {"X": x.reshape(-1, 1), "y": y, "task": "regression"}
    if kind == "clusters":
        X = np.vstack([rng.normal([0, 0], 0.5, (n // 2, 2)), rng.normal([4, 4], 0.5, (n - n // 2, 2))])
        return {"X": X, "y": None, "task": "clustering"}
    raise ValueError(f"Unknown synthetic kind: {kind}")


def ensemble_vote(predictions: Sequence[Sequence[Any]], weights: Optional[Sequence[float]] = None) -> List[Any]:
    """Hard majority vote (or weighted vote) across model predictions.

    predictions: list of per-model label sequences (same length).
    Returns the majority label for each position (ties broken by the
    first model that cast a vote for that label).
    """
    preds = [list(p) for p in predictions]
    n = len(preds[0])
    if weights is None:
        weights = [1.0] * len(preds)
    out = []
    for i in range(n):
        counts: Dict[Any, float] = {}
        order: Dict[Any, int] = {}
        for idx, (p, w) in enumerate(zip(preds, weights)):
            label = p[i]
            if label not in counts:
                counts[label] = 0.0
                order[label] = idx
            counts[label] += w
        best = max(counts, key=lambda k: (counts[k], -order[k]))
        out.append(best)
    return out


__all__ = ["featurize", "make_synthetic", "ensemble_vote"]
