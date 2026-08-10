# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI extended bulk command specs (v4.1) — final groups.

Completes the extended command space:

  mcp          — MCP (model context protocol) helpers
  vector       — vector math helpers
  eval         — evaluation utilities
  quantize     — quantization utilities

Then registers everything with the main bulk registry via
``bulk.rebuild_index()``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import statistics
import string
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from aweai.bulk import spec

# Reuse helpers from bulk engine (private but stable within the package).
from aweai import bulk as _bulk

_num = _bulk._num
_ints = _bulk._ints
_floats = _bulk._floats
_ok = _bulk._ok
_err = _bulk._err
_write = _bulk._write
_read = _bulk._read
_sha256 = _bulk._sha256
_now_iso = _bulk._now_iso
_http_get = _bulk._http_get
_port_open = _bulk._port_open

# ===========================================================================
# MCP group — model context protocol helpers
# ===========================================================================

spec("mcp", "servers", "List known MCP server configs (local store).",
     [], lambda p: _mcp_servers())
spec("mcp", "register", "Register an MCP server.",
     [("name", "github", "Server name"), ("command", "npx", "Launch command"), ("args", "-y @modelcontextprotocol/server-github", "Args")],
     lambda p: _mcp_register(p["name"], p["command"], p["args"]))
spec("mcp", "remove", "Remove an MCP server registration.",
     [("name", "github", "Server name")], lambda p: _mcp_remove(p["name"]))
spec("mcp", "tools", "List tools exposed by a registered server.",
     [("name", "github", "Server name")], lambda p: _mcp_tools(p["name"]))


def _mcp_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/mcp.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _mcp_data() -> Dict[str, Any]:
    p = Path(_mcp_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"servers": {}}
    return {"servers": {}}


def _save_mcp(data: Dict[str, Any]) -> None:
    Path(_mcp_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mcp_servers() -> Dict[str, Any]:
    return _ok(servers=list(_mcp_data()["servers"].values()))


def _mcp_register(name: str, command: str, args: str) -> Dict[str, Any]:
    data = _mcp_data()
    data["servers"][name] = {"name": name, "command": command, "args": args.split(), "registered": _now_iso()}
    _save_mcp(data)
    return _ok(registered=name, command=command)


def _mcp_remove(name: str) -> Dict[str, Any]:
    data = _mcp_data()
    if name in data["servers"]:
        del data["servers"][name]
        _save_mcp(data)
        return _ok(removed=name)
    return _err(f"not found: {name}")


def _mcp_tools(name: str) -> Dict[str, Any]:
    data = _mcp_data()
    s = data["servers"].get(name)
    if s is None:
        return _err(f"not found: {name}")
    return _ok(server=name, tools=[{"name": f"{name}.list_tools", "desc": "list available tools"},
                                   {"name": f"{name}.call_tool", "desc": "call a tool"}])


# ===========================================================================
# VECTOR group — vector math helpers
# ===========================================================================

spec("vector", "add", "Element-wise vector addition.",
     [("a", "1,2,3", "Vector A"), ("b", "4,5,6", "Vector B")],
     lambda p: _vector_op(p["a"], p["b"], "add"))
spec("vector", "sub", "Element-wise vector subtraction.",
     [("a", "4,5,6", "Vector A"), ("b", "1,2,3", "Vector B")],
     lambda p: _vector_op(p["a"], p["b"], "sub"))
spec("vector", "dot", "Dot product.",
     [("a", "1,2,3", "Vector A"), ("b", "4,5,6", "Vector B")],
     lambda p: _vector_op(p["a"], p["b"], "dot"))
spec("vector", "norm", "L2 norm of a vector.",
     [("a", "3,4", "Vector")], lambda p: _vector_norm(p["a"]))
spec("vector", "scale", "Scale a vector by a factor.",
     [("a", "1,2,3", "Vector"), ("factor", 2.0, "Factor")],
     lambda p: _ok(result=[round(x * float(p["factor"]), 4) for x in _floats(p["a"])]))
spec("vector", "avg", "Average of multiple vectors (rows ; separated).",
     [("vectors", "1,2;3,4", "Vectors")], lambda p: _vector_avg(p["vectors"]))
spec("vector", "angle", "Angle (degrees) between two vectors.",
     [("a", "1,0", "Vector A"), ("b", "0,1", "Vector B")],
     lambda p: _vector_angle(p["a"], p["b"]))


def _vector_op(a: str, b: str, op: str) -> Dict[str, Any]:
    va = _floats(a)
    vb = _floats(b)
    if len(va) != len(vb) or not va:
        return _err("vectors must be equal length and non-empty")
    if op == "add":
        return _ok(result=[round(x + y, 4) for x, y in zip(va, vb)])
    if op == "sub":
        return _ok(result=[round(x - y, 4) for x, y in zip(va, vb)])
    return _ok(result=round(sum(x * y for x, y in zip(va, vb)), 4))


def _vector_norm(a: str) -> Dict[str, Any]:
    va = _floats(a)
    return _ok(norm=round(math.sqrt(sum(x * x for x in va)), 4))


def _vector_avg(vectors: str) -> Dict[str, Any]:
    rows = [ _floats(part) for part in vectors.split(";") if part.strip() ]
    if not rows:
        return _err("no vectors")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        return _err("all vectors must be same length")
    avg = [round(sum(r[i] for r in rows) / len(rows), 4) for i in range(n)]
    return _ok(average=avg)


def _vector_angle(a: str, b: str) -> Dict[str, Any]:
    va = _floats(a)
    vb = _floats(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    cos = dot / (na * nb) if na and nb else 0.0
    cos = max(-1.0, min(1.0, cos))
    return _ok(degrees=round(math.degrees(math.acos(cos)), 4), radians=round(math.acos(cos), 4))


# ===========================================================================
# EVAL group — evaluation utilities
# ===========================================================================

spec("eval", "accuracy", "Accuracy of predictions vs labels (comma lists).",
     [("pred", "1,0,1,1", "Predictions"), ("label", "1,0,0,1", "Labels")],
     lambda p: _eval_accuracy(p["pred"], p["label"]))
spec("eval", "precision", "Precision score.",
     [("pred", "1,0,1,1", "Predictions"), ("label", "1,0,0,1", "Labels")],
     lambda p: _eval_precision(p["pred"], p["label"]))
spec("eval", "recall", "Recall score.",
     [("pred", "1,0,1,1", "Predictions"), ("label", "1,0,0,1", "Labels")],
     lambda p: _eval_recall(p["pred"], p["label"]))
spec("eval", "f1", "F1 score.",
     [("pred", "1,0,1,1", "Predictions"), ("label", "1,0,0,1", "Labels")],
     lambda p: _eval_f1(p["pred"], p["label"]))
spec("eval", "confusion", "Confusion matrix counts.",
     [("pred", "1,0,1,1", "Predictions"), ("label", "1,0,0,1", "Labels")],
     lambda p: _eval_confusion(p["pred"], p["label"]))
spec("eval", "mse", "Mean squared error.",
     [("pred", "1,2,3", "Predictions"), ("label", "1,3,3", "Labels")],
     lambda p: _eval_mse(p["pred"], p["label"]))
spec("eval", "mae", "Mean absolute error.",
     [("pred", "1,2,3", "Predictions"), ("label", "1,3,3", "Labels")],
     lambda p: _eval_mae(p["pred"], p["label"]))


def _pairs(pred: str, label: str) -> List[Tuple[float, float]]:
    pv = _floats(pred)
    lv = _floats(label)
    return list(zip(pv, lv))


def _eval_accuracy(pred: str, label: str) -> Dict[str, Any]:
    pairs = _pairs(pred, label)
    if not pairs:
        return _err("no values")
    acc = sum(1 for p, l in pairs if round(p) == round(l)) / len(pairs)
    return _ok(accuracy=round(acc, 4))


def _eval_precision(pred: str, label: str) -> Dict[str, Any]:
    pairs = _pairs(pred, label)
    tp = sum(1 for p, l in pairs if round(p) == 1 and round(l) == 1)
    fp = sum(1 for p, l in pairs if round(p) == 1 and round(l) == 0)
    return _ok(precision=round(tp / (tp + fp), 4) if tp + fp else 0.0)


def _eval_recall(pred: str, label: str) -> Dict[str, Any]:
    pairs = _pairs(pred, label)
    tp = sum(1 for p, l in pairs if round(p) == 1 and round(l) == 1)
    fn = sum(1 for p, l in pairs if round(p) == 0 and round(l) == 1)
    return _ok(recall=round(tp / (tp + fn), 4) if tp + fn else 0.0)


def _eval_f1(pred: str, label: str) -> Dict[str, Any]:
    pr = _eval_precision(pred, label)["precision"]
    re = _eval_recall(pred, label)["recall"]
    return _ok(f1=round(2 * pr * re / (pr + re), 4) if pr + re else 0.0)


def _eval_confusion(pred: str, label: str) -> Dict[str, Any]:
    pairs = _pairs(pred, label)
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for p, l in pairs:
        p = round(p)
        l = round(l)
        if p == 1 and l == 1:
            cm["tp"] += 1
        elif p == 1 and l == 0:
            cm["fp"] += 1
        elif p == 0 and l == 0:
            cm["tn"] += 1
        else:
            cm["fn"] += 1
    return _ok(matrix=cm)


def _eval_mse(pred: str, label: str) -> Dict[str, Any]:
    pairs = _pairs(pred, label)
    if not pairs:
        return _err("no values")
    return _ok(mse=round(sum((p - l) ** 2 for p, l in pairs) / len(pairs), 4))


def _eval_mae(pred: str, label: str) -> Dict[str, Any]:
    pairs = _pairs(pred, label)
    if not pairs:
        return _err("no values")
    return _ok(mae=round(sum(abs(p - l) for p, l in pairs) / len(pairs), 4))


# ===========================================================================
# QUANTIZE group — quantization utilities
# ===========================================================================

spec("quantize", "scale", "Compute scale for symmetric int8 quantization.",
     [("values", "-1,0.5,2", "Values")], lambda p: _quant_scale(p["values"]))
spec("quantize", "to-int8", "Quantize floats to int8 (symmetric).",
     [("values", "-1,0.5,2", "Values")], lambda p: _quant_int8(p["values"]))
spec("quantize", "to-uint8", "Quantize floats to uint8 (asymmetric).",
     [("values", "0,0.5,1", "Values")], lambda p: _quant_uint8(p["values"]))
spec("quantize", "dequant", "Dequantize int8 values back to floats.",
     [("values", "-128,0,127", "Quantized"), ("scale", 0.015625, "Scale")],
     lambda p: _quant_dequant(p["values"], float(p["scale"])))
spec("quantize", "estimate", "Estimate size reduction from quantization.",
     [("params", 1000000, "Parameter count")],
     lambda p: _quant_estimate(int(p["params"])))


def _quant_scale(values: str) -> Dict[str, Any]:
    vals = _floats(values)
    if not vals:
        return _err("no values")
    amax = max(abs(v) for v in vals)
    return _ok(scale=round(amax / 127.0, 8), amax=amax)


def _quant_int8(values: str) -> Dict[str, Any]:
    vals = _floats(values)
    if not vals:
        return _err("no values")
    amax = max(abs(v) for v in vals) or 1.0
    scale = amax / 127.0
    q = [max(-128, min(127, round(v / scale))) for v in vals]
    return _ok(quantized=q, scale=round(scale, 8))


def _quant_uint8(values: str) -> Dict[str, Any]:
    vals = _floats(values)
    if not vals:
        return _err("no values")
    lo, hi = min(vals), max(vals)
    scale = (hi - lo) / 255.0 if hi != lo else 1.0
    q = [max(0, min(255, round((v - lo) / scale))) for v in vals]
    return _ok(quantized=q, scale=round(scale, 8), zero_point=round(lo, 6))


def _quant_dequant(values: str, scale: float) -> Dict[str, Any]:
    vals = _ints(values)
    return _ok(dequantized=[round(v * scale, 6) for v in vals])


def _quant_estimate(params: int) -> Dict[str, Any]:
    return _ok(params=params, fp32_mb=round(params * 4 / 1e6, 3),
               fp16_mb=round(params * 2 / 1e6, 3),
               int8_mb=round(params * 1 / 1e6, 3),
               reduction_pct=round((1 - 1 / 4) * 100, 1))


# ===========================================================================
# Register with the main bulk registry
# ===========================================================================
_bulk.rebuild_index()
