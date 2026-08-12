"""High-level training orchestration.

Everything is model-factory oriented: you tell the factory what data you
have and what model you want; it builds, trains, persists and versions it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aweai.data.loaders import Dataset, load_any
from aweai.data.normalize import normalize_numeric
from aweai.data.split import train_test_split
from aweai.errors import TrainingError
from aweai.hardware import best_device
from aweai.models.base import BaseModel
from aweai.models.registry import create_model, get_model_type_info
from aweai.management.manager import save_model, get_model_path, ModelZooManager


def _prepare_X_y(
    X=None, y=None, data_path: Optional[str] = None,
    text_path: Optional[str] = None, target: Optional[str] = None,
    normalize: Optional[str] = None, seed: Optional[int] = None,
) -> tuple:
    ds: Optional[Dataset] = None
    if data_path:
        ds = load_any(data_path, target_column=target if target else None)
    elif text_path:
        ds = load_any(text_path)
    if ds is not None and X is None:
        if ds.X is not None:
            X = ds.X
        elif ds.texts is not None:
            X = np.array([list(t.encode("utf-8"))[:32] + [0] * max(0, 32 - min(len(t), 32)) for t in ds.texts], dtype=float)
        elif ds.images is not None:
            X = ds.images
        if ds.y is not None and y is None:
            y = ds.y
    if X is None:
        raise TrainingError("No training data provided (pass X/y or data_path/text_path)")
    X = np.asarray(X, dtype=float)
    if y is not None:
        y = np.asarray(y)
    if normalize:
        X, _ = normalize_numeric(X, method=normalize)
    return X, y, ds


def train(
    model_type: str,
    name: str,
    X=None,
    y=None,
    data_path: Optional[str] = None,
    text_path: Optional[str] = None,
    target: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    normalize: Optional[str] = None,
    val_ratio: float = 0.2,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    save: bool = True,
) -> Any:
    """Train a model from scratch and (by default) save it into the zoo."""
    X, y, ds = _prepare_X_y(X=X, y=y, data_path=data_path, text_path=text_path,
                            target=target, normalize=normalize, seed=seed)
    params = dict(params or {})
    info = get_model_type_info(model_type)
    task = info["task"]

    # Auto-infer dimensions from data for models that need them.
    if model_type in ("mlp", "autoencoder", "gan", "cnn", "vision_cnn", "rnn", "lstm", "gru", "object_detector", "segmentation"):
        params.setdefault("input_dim", X.shape[1] if X.ndim == 2 else 1)
    if model_type in ("decision_tree", "random_forest", "naive_bayes", "knn", "svm", "gradient_boosting", "dbscan", "hierarchical"):
        params.setdefault("input_dim", X.shape[1] if X.ndim == 2 else 1)
    if model_type in ("cnn", "vision_cnn"):
        n = X.shape[1]
        h = int(round(n ** 0.5))
        params.setdefault("height", h)
    if model_type in ("mlp", "cnn", "vision_cnn") and y is not None:
        uniq = np.unique(y)
        params.setdefault("output_dim", int(len(uniq)) if len(uniq) > 1 else 1)
        params.setdefault("num_classes", int(len(uniq)) if len(uniq) > 1 else 1)
    if model_type in ("segmentation",):
        params.setdefault("num_classes", int(np.unique(y).size) if y is not None else 2)
    if model_type in ("rnn", "lstm", "gru"):
        params.setdefault("output_dim", X.shape[-1] if X.ndim >= 2 else 1)
        params.setdefault("seq_len", X.shape[1] if X.ndim >= 2 else 4)
    if model_type in ("ts_transformer",):
        params.setdefault("input_dim", X.shape[-1] if X.ndim >= 2 else 1)
        params.setdefault("output_dim", X.shape[-1] if X.ndim >= 2 else 1)
        params.setdefault("max_len", X.shape[1] if X.ndim >= 2 else 8)
    if model_type in ("object_detector",):
        params.setdefault("grid", 4)
        params.setdefault("num_anchors", 2)
        params.setdefault("num_classes", int(np.unique(y).size) if y is not None else 1)
    if model_type in ("gradient_boosting",) and y is not None:
        uniq = np.unique(y)
        params.setdefault("objective", "regression" if y.dtype.kind == "f" else "auto")
    if model_type in ("kmeans",):
        params.setdefault("k", min(3, max(2, int(np.unique(y).size) if y is not None else 3)))
    if model_type in ("transformer",):
        params.setdefault("vocab_size", max(int(X.max()) + 1, 10) if X.size else 10)
        params.setdefault("num_classes", int(np.unique(y).size) if y is not None else 2)

    model = create_model(model_type, **params)
    fit_kwargs = dict(params)
    if model_type in ("ngram",):
        model.fit(X if X is not None else ["hello world"])
    else:
        model.fit(X, y=y, **fit_kwargs)

    if seed is not None:
        np.random.seed(seed)

    if save:
        return save_model(model, name, meta={"model_type": model_type, "task": task})
    return model


def continue_training(name: str, X=None, y=None, data_path: Optional[str] = None,
                      epochs: int = 10, lr: Optional[float] = None, **kw) -> Any:
    """Load an existing model and continue/fine-tune it on new data."""
    manager = ModelZooManager()
    model, meta = manager.load(name)
    X, y, ds = _prepare_X_y(X=X, y=y, data_path=data_path)
    fit_kwargs = dict(model.params)
    if epochs:
        fit_kwargs["epochs"] = epochs
    if lr:
        fit_kwargs["lr"] = lr
    fit_kwargs.update(kw)
    model.fit(X, y=y, **fit_kwargs)
    return manager.save(model, name, meta=meta)


def fit_model(model: BaseModel, X, y=None, **kwargs) -> BaseModel:
    """Low-level: fit any BaseModel instance directly."""
    model.fit(X, y=y, **kwargs)
    return model
