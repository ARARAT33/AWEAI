"""Base class and shared helpers for all from-scratch models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _to_np(X, dtype=float):
    return np.asarray(X, dtype=dtype)


class BaseModel:
    """Common interface: train/predict/save/load/state/export_json."""

    model_type: str = "base"
    is_classifier: bool = False
    is_regressor: bool = False
    is_clusterer: bool = False
    is_generative: bool = False

    def __init__(self, **params):
        self.params: Dict[str, Any] = dict(params)
        self.trained: bool = False
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, list] = {"loss": [], "val_loss": []}
        self.created_at: str = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ------------------------------------------------------------------ API
    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------- helpers
    def _reset_weights(self, *shapes):
        return [np.random.randn(*s).astype(float) * 0.1 for s in shapes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "params": self.params,
            "trained": self.trained,
            "metrics": self.metrics,
            "history": self.history,
            "created_at": self.created_at,
        }

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.to_dict(), "state": self.state_dict()}
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    def load(self, path) -> None:
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        self.params = meta.get("params", {})
        self.trained = meta.get("trained", False)
        self.metrics = meta.get("metrics", {})
        self.history = meta.get("history", {"loss": [], "val_loss": []})
        self.load_state(payload.get("state", {}))

    def export_json(self) -> Dict[str, Any]:
        return self.to_dict()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)
