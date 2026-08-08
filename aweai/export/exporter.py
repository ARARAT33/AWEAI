"""Exporter implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from aweai.errors import ExportError
from aweai.management.manager import get_model_path
from aweai.models.registry import create_model, get_model_type_info
from aweai.utils import read_json, write_json


FORMATS = ["json", "raw", "onnx", "torchscript"]


def export_model(name: str, fmt: str = "json", out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Export a zoo model to the requested format."""
    fmt = (fmt or "json").lower()
    if fmt not in FORMATS:
        raise ExportError(f"Unknown export format: {fmt}. Supported: {FORMATS}")
    root = get_model_path(name)
    payload = read_json(root / "model.json")
    if payload is None:
        raise ExportError(f"Model '{name}' not found in zoo")
    meta = payload.get("meta", {})
    model_type = meta.get("model_type")
    params = dict(meta.get("params", {}))
    model = create_model(model_type, **params)
    model.load_state(payload.get("state", {}))

    out_dir = Path(out_dir) if out_dir else (root / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{name}_v{meta.get('version', 1)}"

    if fmt == "json":
        path = base.with_suffix(".json")
        write_json(path, payload)
    elif fmt == "raw":
        path = base.with_suffix(".npz")
        state = payload.get("state", {})
        arrays = {k: np.asarray(v) for k, v in _flatten_state(state).items()}
        np.savez(path, **arrays)
    elif fmt in ("onnx", "torchscript"):
        if not _HAS_TORCH:
            raise ExportError("torch is required for onnx/torchscript export; pip install torch")
        path = _export_torch(model, fmt, base)
    else:
        raise ExportError(f"Unsupported format {fmt}")
    return {"name": name, "format": fmt, "path": str(path), "size_bytes": path.stat().st_size}


def _flatten_state(state: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                for i, item in enumerate(v):
                    out.update(_flatten_state({f"{key}_{i}": item}))
            else:
                out[key] = v
        elif isinstance(v, dict):
            out.update(_flatten_state(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _export_torch(model, fmt: str, base: Path) -> Path:
    import torch

    # Build a simple torch Module that forwards numpy -> torch -> numpy.
    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x_np = x.detach().cpu().numpy()
            y = self.m.predict(x_np)
            return torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)

    wrapper = _Wrapper(model)
    example = torch.randn(1, model.input_dim) if hasattr(model, "input_dim") else torch.randn(1, 4)
    if fmt == "onnx":
        path = base.with_suffix(".onnx")
        torch.onnx.export(wrapper, example, str(path), input_names=["input"], output_names=["output"])
    else:
        path = base.with_suffix(".pt")
        traced = torch.jit.trace(wrapper, example)
        traced.save(str(path))
    return path
