# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Model zoo manager implementation.

Each model lives in ~/.aweai/models/<name>/
    model.json   — meta + state (BaseModel format) with multi-layer watermarking
    version.json — version counter
    exports/     — ONNX / TorchScript / raw / JSON exports
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aweai.config import ensure_runtime_dirs
from aweai.errors import ModelNotFoundError, ModelError
from aweai.models.base import BaseModel
from aweai.models.registry import create_model, get_model_type_info, list_model_types
from aweai.utils import read_json, safe_filename, write_json
from aweai.watermark import AWEAIWatermarkEngine, DEFAULT_WATERMARK_TEXT


def _model_root() -> Path:
    return ensure_runtime_dirs()["models"]


def get_model_path(name: str) -> Path:
    return _model_root() / safe_filename(name)


class ModelZooManager:
    def __init__(self) -> None:
        self.watermark_engine = AWEAIWatermarkEngine()

    def save(self, model: BaseModel, name: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        name = safe_filename(name)
        root = get_model_path(name)
        root.mkdir(parents=True, exist_ok=True)
        version = int(read_json(root / "version.json", {"v": 0}).get("v", 0)) + 1
        write_json(root / "version.json", {"v": version})

        payload = {"meta": model.to_dict(), "state": model.state_dict()}
        payload["meta"]["name"] = name
        payload["meta"]["version"] = version
        if meta:
            payload["meta"].update(meta)

        # Multi-layer watermarking on payload
        payload = self.watermark_engine.embed_dict(payload, payload=f"MODEL[{name}:v{version}]")
        write_json(root / "model.json", payload)
        return {"name": name, "version": version, "model_type": model.model_type, "path": str(root), "watermarked": True}

    def load(self, name: str) -> Tuple[BaseModel, Dict[str, Any]]:
        root = get_model_path(name)
        payload = read_json(root / "model.json")
        if payload is None:
            raise ModelNotFoundError(f"Model '{name}' not found in zoo")

        # Verify watermark
        watermark_info = self.watermark_engine.verify_dict(payload)
        meta = payload.get("meta", {})
        meta["_watermark_info"] = watermark_info

        model_type = meta.get("model_type")
        if not model_type:
            raise ModelError(f"Model '{name}' has no model_type")
        params = dict(meta.get("params", {}))
        model = create_model(model_type, **params)
        model.load_state(payload.get("state", {}))
        model.metrics = meta.get("metrics", {})
        model.history = meta.get("history", {"loss": [], "val_loss": []})
        model.trained = True
        return model, meta

    def list(self) -> List[Dict[str, Any]]:
        rows = []
        for d in sorted(_model_root().iterdir()) if _model_root().exists() else []:
            if not d.is_dir():
                continue
            payload = read_json(d / "model.json")
            if not payload:
                continue
            meta = payload.get("meta", {})
            rows.append({
                "name": meta.get("name", d.name),
                "model_type": meta.get("model_type", "?"),
                "version": meta.get("version", 0),
                "trained": meta.get("trained", False),
                "metrics": meta.get("metrics", {}),
                "watermark": payload.get("_watermark", DEFAULT_WATERMARK_TEXT),
                "created_at": meta.get("created_at", ""),
                "path": str(d),
            })
        return rows

    def delete(self, name: str) -> bool:
        root = get_model_path(name)
        if not root.exists():
            raise ModelNotFoundError(f"Model '{name}' not found in zoo")
        shutil.rmtree(root)
        return True

    def export(self, name: str, fmt: str = "json", out_dir: Optional[str] = None) -> Dict[str, Any]:
        from aweai.export.exporter import export_model as _export

        return _export(name, fmt=fmt, out_dir=out_dir)

    def import_model(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise ModelNotFoundError(f"Import file not found: {path}")
        payload = read_json(p)
        if payload is None:
            raise ModelError(f"Cannot parse model file: {path}")
        meta = payload.get("meta", {})
        model_type = meta.get("model_type")
        if not model_type or model_type not in list_model_types():
            raise ModelError(f"Unsupported model_type in import: {model_type}")
        params = dict(meta.get("params", {}))
        model = create_model(model_type, **params)
        model.load_state(payload.get("state", {}))
        model.metrics = meta.get("metrics", {})
        model.history = meta.get("history", {"loss": [], "val_loss": []})
        model.trained = True
        target = safe_filename(name or meta.get("name") or p.stem)
        return self.save(model, target, meta={"model_type": model_type})

    def compare(self, names: List[str]) -> Dict[str, Any]:
        out = []
        for n in names:
            model, meta = self.load(n)
            out.append({"name": n, "model_type": meta.get("model_type"), "metrics": meta.get("metrics", {}), "version": meta.get("version", 0)})
        return {"models": out, "count": len(out)}


_singleton: Optional[ModelZooManager] = None


def _manager() -> ModelZooManager:
    global _singleton
    if _singleton is None:
        _singleton = ModelZooManager()
    return _singleton


def save_model(model: BaseModel, name: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _manager().save(model, name, meta=meta)


def load_model(name: str) -> Tuple[BaseModel, Dict[str, Any]]:
    return _manager().load(name)


def list_models() -> List[Dict[str, Any]]:
    return _manager().list()


def delete_model(name: str) -> bool:
    return _manager().delete(name)


def export_model(name: str, fmt: str = "json", out_dir: Optional[str] = None) -> Dict[str, Any]:
    return _manager().export(name, fmt=fmt, out_dir=out_dir)


def import_model(path: str, name: Optional[str] = None) -> Dict[str, Any]:
    return _manager().import_model(path, name=name)


def compare_models(names: List[str]) -> Dict[str, Any]:
    return _manager().compare(names)
