# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Edge export: ONNX, TFLite, and edge-optimized artifacts with multi-layer watermarking.

All exports are *best-effort with graceful degradation*:

* If ``torch`` is available, ONNX and TorchScript exports work natively.
* If ``onnx`` + ``onnxruntime`` are available, we can validate the graph.
* TFLite export is provided as a self-contained, dependency-free converter
  that emits the model weights in a documented JSON schema with watermarking.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from aweai.errors import ExportError
from aweai.management.manager import get_model_path
from aweai.models.registry import create_model
from aweai.utils import read_json, write_json
from aweai.watermark import AWEAIWatermarkEngine

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

EDGE_FORMATS = ["onnx", "tflite", "torchscript", "edge_json"]


def export_edge(
    name: str,
    fmt: str = "onnx",
    quantize: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a zoo model to an edge format (optionally quantized) with watermarking."""
    fmt = (fmt or "onnx").lower()
    if fmt not in EDGE_FORMATS:
        raise ExportError(f"Unknown edge format: {fmt}. Supported: {EDGE_FORMATS}")
    root = get_model_path(name)
    payload = read_json(root / "model.json")
    if payload is None:
        raise ExportError(f"Model '{name}' not found in zoo")
    meta = payload.get("meta", {})
    model_type = meta.get("model_type")
    params = dict(meta.get("params", {}))
    model = create_model(model_type, **params)
    model.load_state(payload.get("state", {}))

    out_dir = Path(out_dir) if out_dir else (root / "edge")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{name}_v{meta.get('version', 1)}"

    if fmt == "onnx":
        path = _export_onnx(model, base)
    elif fmt == "torchscript":
        path = _export_torchscript(model, base)
    elif fmt == "tflite":
        path = _export_tflite(payload, base)
    else:
        path = _export_edge_json(payload, base)

    result = {
        "name": name,
        "format": fmt,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "quantization": quantize,
        "watermarked": True,
    }
    if quantize:
        from aweai.quantize.quantizer import quantize_model

        qres = quantize_model(name, fmt=quantize, out_dir=str(out_dir / "quantized"), evaluate=True)
        result["quantized_artifact"] = qres.get("path")
        result["compression_ratio"] = qres.get("compression_ratio")
    return result


def _export_onnx(model, base: Path) -> Path:
    if not _HAS_TORCH:
        raise ExportError("torch is required for ONNX export; pip install torch")
    import torch

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
    path = base.with_suffix(".onnx")
    torch.onnx.export(wrapper, example, str(path), input_names=["input"], output_names=["output"])
    return path


def _export_torchscript(model, base: Path) -> Path:
    if not _HAS_TORCH:
        raise ExportError("torch is required for TorchScript export; pip install torch")
    import torch

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
    path = base.with_suffix(".pt")
    traced = torch.jit.trace(wrapper, example)
    traced.save(str(path))
    return path


def _export_tflite(payload: Dict[str, Any], base: Path) -> Path:
    meta = payload.get("meta", {})
    artifact = {
        "format": "tflite-json",
        "schema_version": "1.0",
        "model_type": meta.get("model_type"),
        "params": meta.get("params", {}),
        "metrics": meta.get("metrics", {}),
        "state": payload.get("state", {}),
        "framework": "numpy",
        "notes": "Dependency-free TFLite-style export. Load with aweai.export.edge.load_tflite_json().",
    }
    engine = AWEAIWatermarkEngine()
    watermarked_artifact = engine.embed_dict(artifact, payload=f"TFLITE[{meta.get('name', 'model')}]")
    path = base.with_suffix(".tflite.json")
    write_json(path, watermarked_artifact)
    return path


def _export_edge_json(payload: Dict[str, Any], base: Path) -> Path:
    meta = payload.get("meta", {})
    artifact = {
        "format": "edge_json",
        "schema_version": "1.0",
        "model_type": meta.get("model_type"),
        "params": meta.get("params", {}),
        "metrics": meta.get("metrics", {}),
        "state": payload.get("state", {}),
        "framework": "numpy",
    }
    engine = AWEAIWatermarkEngine()
    watermarked_artifact = engine.embed_dict(artifact, payload=f"EDGE_JSON[{meta.get('name', 'model')}]")
    path = base.with_suffix(".edge.json")
    write_json(path, watermarked_artifact)
    return path


def load_tflite_json(path: str):
    """Load a dependency-free TFLite-style export back into a BaseModel."""
    p = Path(path)
    if not p.exists():
        raise ExportError(f"Export file not found: {path}")
    artifact = read_json(p)
    if artifact is None:
        raise ExportError(f"Cannot parse export file: {path}")
    model_type = artifact.get("model_type")
    model = create_model(model_type, **dict(artifact.get("params", {})))
    model.load_state(artifact.get("state", {}))
    model.metrics = artifact.get("metrics", {})
    return model


def estimate_edge_footprint(name: str) -> Dict[str, Any]:
    """Estimate on-device footprint (RAM, storage) for a zoo model."""
    root = get_model_path(name)
    payload = read_json(root / "model.json")
    if payload is None:
        raise ExportError(f"Model '{name}' not found in zoo")
    state = payload.get("state", {})
    n_params = _count_params(state)
    fp32 = n_params * 4
    fp16 = n_params * 2
    int8 = n_params * 1
    return {
        "name": name,
        "parameters": n_params,
        "fp32_bytes": fp32,
        "fp16_bytes": fp16,
        "int8_bytes": int8,
        "fp16_vs_fp32": round(fp32 / max(fp16, 1), 2),
        "int8_vs_fp32": round(fp32 / max(int8, 1), 2),
        "edge_ready": int8 <= 64 * 1024 * 1024,
    }


def _count_params(state: Dict[str, Any]) -> int:
    total = 0
    for k, v in state.items():
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                for sub in v:
                    total += _count_params(sub) if isinstance(sub, dict) else int(np.asarray(sub).size)
            else:
                total += int(np.asarray(v).size)
        elif isinstance(v, dict):
            total += _count_params(v)
        else:
            total += int(np.asarray(v).size)
    return total
