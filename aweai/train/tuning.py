"""Hyperparameter tuning: grid search + random search over model params."""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.registry import create_model


def _score(model, X, y, task: str) -> float:
    pred = model.predict(X)
    if task == "classification":
        return float(np.mean(np.asarray(pred) == np.asarray(y)))
    if task == "regression":
        return -float(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))
    return -float(model.metrics.get("inertia", 0.0))


def grid_search(model_type: str, X, y=None, param_grid: Optional[Dict[str, List]] = None,
                task: Optional[str] = None, n_trials: Optional[int] = None) -> Dict[str, Any]:
    """Grid search over parameter combinations; returns best result."""
    from aweai.models.registry import MODEL_TYPES, get_model_type_info

    param_grid = param_grid or {"epochs": [5, 10]}
    task = task or get_model_type_info(model_type)["task"]
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    if n_trials:
        combos = combos[:n_trials]
    best: Optional[Dict[str, Any]] = None
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            model = create_model(model_type, **params)
            model.fit(X, y=y, **params)
            score = _score(model, X, y, task)
            row = {"params": params, "score": score, "metrics": dict(model.metrics)}
            if best is None or score > best["score"]:
                best = row
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Grid search failed: no valid combination")
    return best


def random_search(model_type: str, X, y=None, param_grid: Optional[Dict[str, List]] = None,
                  n_trials: int = 5, task: Optional[str] = None) -> Dict[str, Any]:
    from aweai.models.registry import get_model_type_info

    param_grid = param_grid or {"epochs": [5, 10, 15]}
    task = task or get_model_type_info(model_type)["task"]
    best: Optional[Dict[str, Any]] = None
    for _ in range(n_trials):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        try:
            model = create_model(model_type, **params)
            model.fit(X, y=y, **params)
            score = _score(model, X, y, task)
            row = {"params": params, "score": score, "metrics": dict(model.metrics)}
            if best is None or score > best["score"]:
                best = row
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Random search failed: no valid combination")
    return best


def tune(model_type: str, X, y=None, method: str = "grid", **kwargs) -> Dict[str, Any]:
    if method == "random":
        return random_search(model_type, X, y=y, **kwargs)
    return grid_search(model_type, X, y=y, **kwargs)
