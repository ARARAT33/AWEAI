"""Train/test splitting."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from aweai.errors import DataError


def split_by_ratio(n: int, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise DataError("Cannot split an empty dataset")
    if not 0.0 < ratio < 1.0:
        raise DataError(f"ratio must be in (0,1), got {ratio}")
    idx = np.random.permutation(n)
    cut = int(n * ratio)
    return idx[:cut], idx[cut:]


def train_test_split(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    texts=None,
    ratio: float = 0.8,
    seed: Optional[int] = None,
) -> dict:
    if seed is not None:
        np.random.seed(seed)
    if X is not None:
        n = len(X)
    elif texts is not None:
        n = len(texts)
    else:
        raise DataError("Provide X or texts to split")
    tr, te = split_by_ratio(n, ratio)
    out: dict = {}
    if X is not None:
        out["X_train"], out["X_test"] = X[tr], X[te]
    if y is not None:
        out["y_train"], out["y_test"] = y[tr], y[te]
    if texts is not None:
        out["texts_train"] = [texts[i] for i in tr]
        out["texts_test"] = [texts[i] for i in te]
    return out
