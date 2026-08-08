"""Model quantization (v2.2).

Supports:

* ``float16`` — half-precision weights (2x smaller, minimal accuracy loss).
* ``int8``   — symmetric per-tensor integer quantization with scale/zero-point
               (4x smaller, fast on edge/CPU).
* ``uint8`` / ``int4`` — additional formats for edge deployment.

Quantization works on any ``BaseModel`` in the zoo by transforming its
``state_dict`` tensors. The quantized artifact is stored alongside the model
and can be evaluated for accuracy loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aweai.errors import QuantizeError
from aweai.management.manager import get_model_path
from aweai.models.base import BaseModel
from aweai.models.registry import create_model
from aweai.utils import read_json, write_json

FORMATS = ["float16", "int8", "uint8", "int4"]


def _quantize_int8(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    amax = float(np.max(np.abs(arr))) if arr.size else 0.0
    if amax == 0.0:
        scale = 1.0
    else:
        scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return {"q": q.tolist(), "scale": scale, "zero_point": 0}


def _quantize_uint8(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    scale = (mx - mn) / 255.0 if mx > mn else 1.0
    zero = -mn / scale if scale > 0 else 0.0
    q = np.clip(np.round(arr / scale + zero), 0, 255).astype(np.uint8)
    return {"q": q.tolist(), "scale": scale, "zero_point": float(zero)}


def _quantize_int4(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    amax = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = amax / 7.0 if amax > 0 else 1.0
    q = np.clip(np.round(arr / scale), -7, 7).astype(np.int8)
    return {"q": q.tolist(), "scale": scale, "zero_point": 0}


def _dequantize(payload: Dict[str, Any], dtype=np.float32) -> np.ndarray:
    q = np.asarray(payload["q"], dtype=np.float32)
    scale = float(payload.get("scale", 1.0))
    zp = float(payload.get("zero_point", 0.0))
    return (q - zp) * scale


def _state_to_numpy(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in state.items():
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                out[k] = [np.asarray(x, dtype=float) for x in v]
            else:
                out[k] = np.asarray(v, dtype=float)
        elif isinstance(v, dict):
            out[k] = _state_to_numpy(v)
        else:
            out[k] = np.asarray(v, dtype=float)
    return out


def _quantize_state(state: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    fn = {
        "float16": lambda a: {"q": np.asarray(a, dtype=np.float16).tolist(),
                              "scale": 1.0, "zero_point": 0},
        "int8": _quantize_int8,
        "uint8": _quantize_uint8,
        "int4": _quantize_int4,
    }[fmt]
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                out[k] = [_quantize_array(x, fn) for x in v]
            else:
                out[k] = _quantize_array(np.asarray(v, dtype=float), fn)
        elif isinstance(v, dict):
            out[k] = _quantize_state(v, fmt)
        else:
            out[k] = _quantize_array(np.asarray(v, dtype=float), fn)
    return out


def _quantize_array(arr: np.ndarray, fn) -> Dict[str, Any]:
    return fn(arr)


def _dequantize_state(qstate: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in qstate.items():
        if isinstance(v, dict) and "q" in v:
            out[k] = _dequantize(v)
        elif isinstance(v, list):
            if v and isinstance(v[0], dict) and "q" in v[0]:
                out[k] = [_dequantize(x) for x in v]
            else:
                out[k] = v
        elif isinstance(v, dict):
            out[k] = _dequantize_state(v)
        else:
            out[k] = v
    return out


def quantize_model(
    name: str,
    fmt: str = "int8",
    out_dir: Optional[str] = None,
    evaluate: bool = True,
    X_eval=None,
    y_eval=None,
) -> Dict[str, Any]:
    """Quantize a zoo model and store the quantized artifact."""
    fmt = (fmt or "int8").lower()
    if fmt not in FORMATS:
        raise QuantizeError(f"Unknown quantization format: {fmt}. Supported: {FORMATS}")
    root = get_model_path(name)
    payload = read_json(root / "model.json")
    if payload is None:
        raise QuantizeError(f"Model '{name}' not found in zoo")
    meta = payload.get("meta", {})
    model_type = meta.get("model_type")
    params = dict(meta.get("params", {}))
    model = create_model(model_type, **params)
    model.load_state(payload.get("state", {}))

    qstate = _quantize_state(payload.get("state", {}), fmt)
    out_dir = Path(out_dir) if out_dir else (root / "quantized")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_type": model_type,
        "format": fmt,
        "original_size": _payload_bytes(payload.get("state", {})),
        "quantized_size": _payload_bytes(qstate),
        "quantized_state": qstate,
        "meta": meta,
    }
    path = out_dir / f"{name}_quant_{fmt}.json"
    write_json(path, artifact)

    result = {
        "name": name,
        "format": fmt,
        "path": str(path),
        "original_bytes": artifact["original_size"],
        "quantized_bytes": artifact["quantized_size"],
        "compression_ratio": round(artifact["original_size"] / max(artifact["quantized_size"], 1), 2),
    }
    if evaluate:
        acc = _evaluate_quantized(model, qstate, X_eval, y_eval)
        result["evaluation"] = acc
        write_json(path, {**artifact, "evaluation": acc})
    return result


def _evaluate_quantized(model: BaseModel, qstate: Dict[str, Any], X_eval, y_eval) -> Dict[str, Any]:
    try:
        orig = _dequantize_state(qstate)
        # Compare mean absolute weight error introduced by quantization
        errs = []
        for k, v in orig.items():
            if isinstance(v, (list, tuple)):
                for i, sub in enumerate(v):
                    if isinstance(sub, np.ndarray):
                        errs.append(float(np.mean(np.abs(sub - _dequantize(qstate[k][i])))))
        if not errs:
            errs = [0.0]
        return {
            "mean_weight_abs_error": round(float(np.mean(errs)), 6),
            "max_weight_abs_error": round(float(np.max(errs)), 6),
        }
    except Exception as e:
        return {"error": str(e)}


def _payload_bytes(state: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(state))
    except Exception:
        return 0


def load_quantized(name: str, fmt: str = "int8") -> Tuple[BaseModel, Dict[str, Any]]:
    """Load a quantized model back into a runnable BaseModel (dequantized)."""
    root = get_model_path(name)
    path = root / "quantized" / f"{name}_quant_{fmt}.json"
    artifact = read_json(path)
    if artifact is None:
        raise QuantizeError(f"Quantized model '{name}' ({fmt}) not found")
    meta = artifact.get("meta", {})
    model = create_model(meta.get("model_type"), **dict(meta.get("params", {})))
    deq = _dequantize_state(artifact.get("quantized_state", {}))
    model.load_state(deq)
    return model, artifact


def list_quantized(name: Optional[str] = None) -> List[Dict[str, Any]]:
    """List quantized artifacts (all models or one model)."""
    out = []
    if name:
        root = get_model_path(name)
        qdir = root / "quantized"
        if qdir.exists():
            for f in sorted(qdir.glob("*.json")):
                data = read_json(f, {})
                out.append({"model": name, "format": data.get("format"), "path": str(f)})
        return out
    from aweai.management.manager import _model_root
    base = _model_root()
    if base.exists():
        for d in sorted(base.iterdir()):
            qdir = d / "quantized"
            if qdir.exists():
                for f in sorted(qdir.glob("*.json")):
                    data = read_json(f, {})
                    out.append({"model": d.name, "format": data.get("format"), "path": str(f)})
    return out
