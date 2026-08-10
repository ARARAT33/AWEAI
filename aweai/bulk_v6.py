# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI v4.1 bulk command specs (batch 2).

Adds even more declarative commands across new groups:

  vision    - image utilities: resize params, color channels, aspect, EXIF-ish
  audio     - audio utilities: duration estimate, sample rate, freq helpers
  video     - video utilities: frame count, bitrate estimate
  dataset   - dataset lifecycle: version, split, stats, merge, export
  feature   - feature engineering: transforms, selection, binning
  quant     - quantization utilities: scale, int8/uint8, dequantize, estimate
  deploy    - deployment: targets, manifest, rollback, env-check
  api       - API helpers: endpoint check, schema, mock, curl
  dataops   - data operations: pipelines, jobs, lineage
  crypto2   - more crypto: base64, rot13, caesar, url-encode, uuid
  geo       - geo helpers: distance, bbox, dms
  stats2    - statistics: t-test, chi2, entropy, mutual info
  mcp2      - MCP registry helpers
  agent2    - more agent orchestration: spawn, stop, list-runs
  workflow2 - more workflow: trigger, retry, status
  monitor2  - more monitoring: cpu, mem, load, disk, net

Every spec follows the same declarative shape used by :mod:`aweai.bulk`
(name, help, params, fn) and is appended to the main registry.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import statistics
import string
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


def _store_path(name: str) -> str:
    p = Path(os.path.expanduser("~/.aweai")) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _load_store(path: str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_store(path: str, obj: Any) -> Dict[str, Any]:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return _ok(path=path)


def _run(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"error: {e}"


# ===========================================================================
# VISION group - image utilities
# ===========================================================================


def _aspect(w: int, h: int) -> str:
    g = math.gcd(w, h) or 1
    return f"{w // g}:{h // g}"


spec("vision", "aspect", "Compute aspect ratio of an image size.",
     [("width", 1920, "Width px"), ("height", 1080, "Height px")],
     lambda p: _ok(width=int(p["width"]), height=int(p["height"]), aspect=_aspect(int(p["width"]), int(p["height"]))))

spec("vision", "channels", "Return expected channels per color mode.",
     [("mode", "rgb", "rgb|rgba|grayscale|l|p")],
     lambda p: _ok(mode=p["mode"], channels={"rgb": 3, "rgba": 4, "grayscale": 1, "l": 1, "p": 1}.get(p["mode"], 3)))

spec("vision", "resize", "Compute resized dimensions preserving aspect.",
     [("width", 1920, "Width px"), ("height", 1080, "Height px"), ("max_size", 512, "Max side px")],
     lambda p: _ok(resized=({"width": int(p["max_size"]), "height": int(int(p["height"]) * int(p["max_size"]) / int(p["width"]))}
                            if int(p["width"]) >= int(p["height"]) else
                            {"width": int(int(p["width"]) * int(p["max_size"]) / int(p["height"])), "height": int(p["max_size"])})))

spec("vision", "megapixels", "Compute megapixels from dimensions.",
     [("width", 1920, "Width px"), ("height", 1080, "Height px")],
     lambda p: _ok(megapixels=round(int(p["width"]) * int(p["height"]) / 1e6, 3)))

spec("vision", "grayscale", "Convert RGB triplet to grayscale luminance.",
     [("rgb", "255,128,64", "Comma-separated r,g,b")],
     lambda p: _ok(gray=round(0.299 * _floats(p["rgb"])[0] + 0.587 * _floats(p["rgb"])[1] + 0.114 * _floats(p["rgb"])[2], 2)))

spec("vision", "invert", "Invert RGB triplet.",
     [("rgb", "255,128,64", "Comma-separated r,g,b")],
     lambda p: _ok(inverted=[round(255 - v, 1) for v in _floats(p["rgb"])]))

spec("vision", "brightness", "Scale RGB triplet brightness.",
     [("rgb", "100,100,100", "Comma-separated r,g,b"), ("factor", 1.5, "Brightness factor")],
     lambda p: _ok(brightened=[max(0, min(255, round(v * p["factor"], 1))) for v in _floats(p["rgb"])]))

spec("vision", "contrast", "Apply contrast factor to RGB triplet.",
     [("rgb", "100,100,100", "Comma-separated r,g,b"), ("factor", 1.5, "Contrast factor")],
     lambda p: _ok(contrasted=[max(0, min(255, round((v - 128) * p["factor"] + 128, 1))) for v in _floats(p["rgb"])]))

spec("vision", "hex2rgb", "Convert hex color to RGB.",
     [("hex", "#ff8000", "Hex color")],
     lambda p: _ok(rgb=[int(p["hex"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]))

spec("vision", "rgb2hex", "Convert RGB triplet to hex.",
     [("rgb", "255,128,0", "Comma-separated r,g,b")],
     lambda p: _ok(hex="#" + "".join(f"{int(v):02x}" for v in _floats(p["rgb"]))))

spec("vision", "format-info", "Show typical image format info.",
     [("format", "png", "png|jpg|webp|gif|bmp")],
     lambda p: _ok(format=p["format"], lossy=p["format"] in ("jpg", "webp"),
                   supports_alpha=p["format"] in ("png", "webp", "gif", "bmp")))


# ===========================================================================
# AUDIO group - audio utilities
# ===========================================================================


def _audio_bytes_duration(size_mb: float, bitrate_kbps: float) -> float:
    return round(size_mb * 8 * 1024 / bitrate_kbps, 2)


def _nyquist(sample_rate: int) -> float:
    return sample_rate / 2.0


spec("audio", "duration", "Estimate audio duration from file size and bitrate.",
     [("size_mb", 4.0, "File size MB"), ("bitrate_kbps", 128.0, "Bitrate kbps")],
     lambda p: _ok(seconds=_audio_bytes_duration(p["size_mb"], p["bitrate_kbps"])))

spec("audio", "nyquist", "Nyquist frequency for a sample rate.",
     [("sample_rate", 44100, "Sample rate Hz")],
     lambda p: _ok(nyquist_hz=_nyquist(int(p["sample_rate"]))))

spec("audio", "sample-size", "Bytes per second for PCM audio.",
     [("sample_rate", 44100, "Sample rate Hz"), ("bits", 16, "Bits per sample"), ("channels", 2, "Channels")],
     lambda p: _ok(bytes_per_sec=int(p["sample_rate"]) * int(p["bits"]) // 8 * int(p["channels"])))

spec("audio", "freq", "Convert note name to frequency (A4=440).",
     [("note", "A4", "Note like A4, C5")],
     lambda p: _ok(freq=round(440.0 * 2 ** ((int(p["note"][1:]) - 4) + (ord(p["note"][0].upper()) - ord("A")) / 12.0), 2)))

spec("audio", "loudness", "Approximate loudness (dB) from linear amplitude.",
     [("amplitude", 0.5, "Amplitude 0-1")],
     lambda p: _ok(db=round(20 * math.log10(max(1e-6, p["amplitude"])), 2)))

spec("audio", "silence", "Estimate silence duration in a clip.",
     [("duration", 120, "Clip seconds"), ("silence_ratio", 0.1, "Silence ratio 0-1")],
     lambda p: _ok(silence_seconds=round(p["duration"] * p["silence_ratio"], 2)))


# ===========================================================================
# VIDEO group - video utilities
# ===========================================================================


def _video_frames(duration_s: float, fps: float) -> int:
    return int(duration_s * fps)


def _video_bitrate(resolution: str, fps: float, codec: str) -> int:
    base = {"720p": 2500, "1080p": 5000, "4k": 16000}.get(resolution, 2500)
    return int(base * (fps / 30.0) * (1.2 if codec == "h264" else 0.6))


spec("video", "frames", "Estimate total frames.",
     [("duration", 60.0, "Duration seconds"), ("fps", 30.0, "Frames per second")],
     lambda p: _ok(frames=_video_frames(p["duration"], p["fps"])))

spec("video", "bitrate", "Estimate video bitrate for resolution/codec.",
     [("resolution", "1080p", "720p|1080p|4k"), ("fps", 30.0, "FPS"), ("codec", "h264", "h264|hevc")],
     lambda p: _ok(bitrate_kbps=_video_bitrate(p["resolution"], p["fps"], p["codec"])))

spec("video", "size", "Estimate file size from bitrate and duration.",
     [("bitrate_kbps", 5000, "Bitrate kbps"), ("duration", 60.0, "Duration seconds")],
     lambda p: _ok(size_mb=round(int(p["bitrate_kbps"]) * p["duration"] / 8 / 1024, 2)))

spec("video", "storage", "Estimate storage for a video collection.",
     [("size_mb", 100.0, "Per-file MB"), ("count", 100, "File count")],
     lambda p: _ok(total_gb=round(p["size_mb"] * int(p["count"]) / 1024, 2)))


# ===========================================================================
# DATASET group - dataset lifecycle
# ===========================================================================


def _dataset_file() -> str:
    return _store_path("datasets.json")


def _datasets() -> Dict[str, Any]:
    return _load_store(_dataset_file(), {}) or {}


def _dataset_rows(data: str) -> List[List[str]]:
    return [row.split(",") for row in data.strip().splitlines() if row.strip()]


spec("dataset", "create", "Create a dataset definition from CSV data.",
     [("name", "my-ds", "Dataset name"), ("data", "a,b\n1,2\n3,4", "CSV data")],
     lambda p: _ok(**_save_store(_dataset_file(), {**_datasets(), p["name"]: {
         "rows": len(_dataset_rows(p["data"])), "cols": len(_dataset_rows(p["data"])[0]) if _dataset_rows(p["data"]) else 0,
         "created": _now_iso(), "data": p["data"]}})))

spec("dataset", "list", "List datasets.",
     [],
     lambda p: _ok(datasets=[{"name": k, "rows": v.get("rows"), "cols": v.get("cols")} for k, v in _datasets().items()]))

spec("dataset", "show", "Show dataset summary.",
     [("name", "my-ds", "Dataset name")],
     lambda p: _ok(**_datasets().get(p["name"], {})) if p["name"] in _datasets() else _err(f"dataset '{p['name']}' not found"))

spec("dataset", "remove", "Remove a dataset.",
     [("name", "my-ds", "Dataset name")],
     lambda p: _ok(removed=p["name"], **(_save_store(_dataset_file(), {k: v for k, v in _datasets().items() if k != p["name"]}))) if p["name"] in _datasets() else _err("not found"))

spec("dataset", "split", "Split CSV data into train/test by ratio.",
     [("data", "1,2\n3,4\n5,6\n7,8\n9,10", "CSV data"), ("ratio", 0.8, "Train ratio")],
     lambda p: _ok(train_rows=int(len(_dataset_rows(p["data"])) * p["ratio"]), test_rows=len(_dataset_rows(p["data"])) - int(len(_dataset_rows(p["data"])) * p["ratio"])))

spec("dataset", "stats", "Basic stats of a dataset.",
     [("data", "1,2,3,4,5,6,7,8,9,10", "CSV data")],
     lambda p: _ok(rows=len(_dataset_rows(p["data"])),
                   numeric_count=sum(1 for r in _dataset_rows(p["data"]) for c in r if c.strip().replace(".", "").isdigit()),
                   sample=_dataset_rows(p["data"])[:2]))

spec("dataset", "merge", "Merge two CSV datasets vertically.",
     [("a", "h\n1\n2", "CSV A"), ("b", "h\n3\n4", "CSV B")],
     lambda p: _ok(merged=p["a"].strip() + "\n" + "\n".join(_dataset_rows(p["b"])[1:])))

spec("dataset", "export", "Export dataset to JSON file.",
     [("name", "my-ds", "Dataset name"), ("out", "exports/dataset.json", "Output path")],
     lambda p: _ok(**_write(p["out"], json.dumps(_datasets().get(p["name"], {}), indent=2))) if p["name"] in _datasets() else _err("not found"))

spec("dataset", "version", "Create a versioned snapshot of a dataset.",
     [("name", "my-ds", "Dataset name"), ("version", "v1", "Version tag")],
     lambda p: _ok(version=p["version"], **(_save_store(_dataset_file() + f".{p['version']}", _datasets().get(p["name"], {})))) if p["name"] in _datasets() else _err("not found"))


# ===========================================================================
# FEATURE group - feature engineering
# ===========================================================================


def _bin_values(vals: List[float], bins: int) -> List[str]:
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return ["same"] * len(vals)
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    return [f"bin{i}" for v in vals for i in range(bins) if edges[i] <= v < edges[i + 1]] + (["bin0"] if False else [])


spec("feature", "bin", "Bin numeric values into buckets.",
     [("values", "1,2,3,4,5,6,7,8,9,10", "Comma-separated numbers"), ("bins", 5, "Bin count")],
     lambda p: _ok(bins=_bin_values(_floats(p["values"]), int(p["bins"]))[: len(_floats(p["values"]))]))

spec("feature", "onehot", "One-hot encode labels.",
     [("labels", "cat,dog,cat,bird", "Comma-separated labels")],
     lambda p: _ok(classes=sorted(set(p["labels"].split(","))),
                   encoded=[{c: 1 if l == c else 0 for c in sorted(set(p["labels"].split(",")))} for l in p["labels"].split(",")]))

spec("feature", "scaling", "Choose scaling method based on outliers.",
     [("values", "1,2,3,4,100", "Comma-separated numbers")],
     lambda p: _ok(recommended="robust" if any(abs(v - statistics.mean(_floats(p["values"]))) > 2 * (statistics.stdev(_floats(p["values"])) or 1) for v in _floats(p["values"])) else "standard"))

spec("feature", "missing-fill", "Suggest fill strategy for missing values.",
     [("col_type", "numeric", "numeric|categorical")],
     lambda p: _ok(strategy="mean/median" if p["col_type"] == "numeric" else "mode", note="impute before scaling"))

spec("feature", "select", "Select top features by simple variance threshold.",
     [("values", "1,1,1,2,2,2,3,3,3,4", "Comma-separated numbers"), ("threshold", 0.5, "Variance threshold")],
     lambda p: _ok(variance=round(statistics.variance(_floats(p["values"])), 4),
                   keep=statistics.variance(_floats(p["values"])) >= p["threshold"]))

spec("feature", "skew-fix", "Suggest transform for skewed data.",
     [("skew", 2.5, "Skewness value")],
     lambda p: _ok(transform="log1p" if abs(p["skew"]) > 1 else ("sqrt" if abs(p["skew"]) > 0.5 else "none")))

spec("feature", "date-parts", "Split ISO date into parts.",
     [("date", "2026-08-10", "ISO date")],
     lambda p: _ok(parts={"year": int(p["date"][:4]), "month": int(p["date"][5:7]), "day": int(p["date"][8:10])}))

spec("feature", "text-length", "Add text length feature suggestions.",
     [("text", "hello world", "Text")],
     lambda p: _ok(char_count=len(p["text"]), word_count=len(p["text"].split()), token_estimate=_num(p["text"]) and max(1, len(p["text"]) // 4)))


# ===========================================================================
# QUANT group - quantization utilities (alias of quantize)
# ===========================================================================


def _q_scale(vals: List[float]) -> Dict[str, Any]:
    amax = max(abs(v) for v in vals) or 1.0
    return _ok(scale=round(amax / 127.0, 8), amax=amax)


def _q_int8(vals: List[float]) -> Dict[str, Any]:
    amax = max(abs(v) for v in vals) or 1.0
    scale = amax / 127.0
    return _ok(quantized=[max(-128, min(127, round(v / scale))) for v in vals], scale=round(scale, 8))


def _q_uint8(vals: List[float]) -> Dict[str, Any]:
    lo, hi = min(vals), max(vals)
    scale = (hi - lo) / 255.0 if hi != lo else 1.0
    return _ok(quantized=[max(0, min(255, round((v - lo) / scale))) for v in vals], scale=round(scale, 8), zero_point=round(lo, 6))


def _q_dequant(vals: List[int], scale: float) -> Dict[str, Any]:
    return _ok(dequantized=[round(v * scale, 6) for v in vals])


def _q_estimate(params: int) -> Dict[str, Any]:
    return _ok(params=params, fp32_mb=round(params * 4 / 1e6, 3), fp16_mb=round(params * 2 / 1e6, 3),
               int8_mb=round(params * 1 / 1e6, 3), reduction_pct=round((1 - 1 / 4) * 100, 1))


spec("quant", "scale", "Compute symmetric int8 quantization scale.",
     [("values", "1.5,-2.0,3.5", "Comma-separated floats")],
     lambda p: _q_scale(_floats(p["values"])))

spec("quant", "to-int8", "Quantize floats to int8 (symmetric).",
     [("values", "1.5,-2.0,3.5", "Comma-separated floats")],
     lambda p: _q_int8(_floats(p["values"])))

spec("quant", "to-uint8", "Quantize floats to uint8 (asymmetric).",
     [("values", "1.5,-2.0,3.5", "Comma-separated floats")],
     lambda p: _q_uint8(_floats(p["values"])))

spec("quant", "dequant", "Dequantize int8 values back to floats.",
     [("values", "48,-64,112", "Comma-separated ints"), ("scale", 0.02755906, "Scale")],
     lambda p: _q_dequant(_ints(p["values"]), p["scale"]))

spec("quant", "estimate", "Estimate memory reduction from quantization.",
     [("params", 1000000, "Parameter count")],
     lambda p: _q_estimate(int(p["params"])))


# ===========================================================================
# DEPLOY group - deployment helpers
# ===========================================================================


def _deploy_targets() -> List[str]:
    return ["local", "docker", "kubernetes", "aws", "gcp", "azure", "edge", "serverless"]


spec("deploy", "targets", "List supported deployment targets.",
     [],
     lambda p: _ok(targets=_deploy_targets()))

spec("deploy", "manifest", "Generate a minimal deployment manifest.",
     [("name", "aweai-app", "App name"), ("image", "python:3.11", "Container image"), ("port", 8000, "Port")],
     lambda p: _ok(manifest={"name": p["name"], "image": p["image"], "port": int(p["port"]),
                             "replicas": 1, "env": [], "healthcheck": f"/health"}))

spec("deploy", "rollback", "Plan a rollback strategy.",
     [("current", "v1.2", "Current version"), ("previous", "v1.1", "Previous version")],
     lambda p: _ok(plan=[f"tag {p['previous']} as safe", f"rollback image to {p['previous']}", "run smoke tests", "verify metrics"]))

spec("deploy", "env-check", "Check required env vars for deployment.",
     [("required", "API_KEY,DATABASE_URL", "Comma-separated required vars")],
     lambda p: _ok(missing=[v for v in p["required"].split(",") if v.strip() and not os.environ.get(v.strip())],
                   present=[v for v in p["required"].split(",") if v.strip() and os.environ.get(v.strip())]))

spec("deploy", "compose", "Generate docker-compose snippet.",
     [("name", "aweai-app", "Service name"), ("image", "python:3.11", "Image"), ("port", 8000, "Port")],
     lambda p: _ok(compose=f"services:\n  {p['name']}:\n    image: {p['image']}\n    ports:\n      - \"{p['port']}:{p['port']}\"\n    restart: unless-stopped"))


# ===========================================================================
# API group - API helpers
# ===========================================================================


def _api_status(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AWEAI-CLI/4.1"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


spec("api", "check", "Check if an API endpoint is reachable.",
     [("url", "https://api.github.com", "Endpoint URL")],
     lambda p: _ok(url=p["url"], status=_api_status(p["url"]), reachable=_api_status(p["url"]) < 500))

spec("api", "schema", "Generate a JSON schema from field:type pairs.",
     [("name", "User", "Schema name"), ("fields", "id:int,name:string", "Comma-separated field:type")],
     lambda p: _ok(schema={"$schema": "http://json-schema.org/draft-07/schema#", "title": p["name"],
                           "type": "object", "properties": {f.split(":")[0]: {"type": f.split(":")[1] if len(f.split(":")) > 1 else "string"}
                                                            for f in p["fields"].split(",") if ":" in f},
                           "required": [f.split(":")[0] for f in p["fields"].split(",") if ":" in f]}))

spec("api", "curl", "Build a curl command from parameters.",
     [("method", "GET", "HTTP method"), ("url", "https://api.example.com/data", "URL"),
      ("data", "", "JSON body (optional)"), ("token", "", "Bearer token (optional)")],
     lambda p: _ok(curl="curl -s -X " + p["method"] + " " + ("-H \"Authorization: Bearer " + p["token"] + "\" " if p["token"] else "") +
                   ("-H \"Content-Type: application/json\" -d '" + p["data"] + "' " if p["data"] else "") + "'" + p["url"] + "'"))

spec("api", "mock", "Generate a mock JSON response.",
     [("fields", "id:1,name:test", "Comma-separated field:value")],
     lambda p: _ok(mock={kv.split(":")[0]: kv.split(":", 1)[1] for kv in p["fields"].split(",") if ":" in kv}))

spec("api", "endpoint-list", "List common REST endpoints for a resource.",
     [("resource", "users", "Resource name")],
     lambda p: _ok(endpoints=[f"GET /{p['resource']}", f"GET /{p['resource']}/{{id}}", f"POST /{p['resource']}",
                              f"PUT /{p['resource']}/{{id}}", f"DELETE /{p['resource']}/{{id}}"]))

spec("api", "health", "Build a health-check URL.",
     [("base", "https://api.example.com", "Base URL"), ("path", "/health", "Health path")],
     lambda p: _ok(url=p["base"].rstrip("/") + p["path"]))


# ===========================================================================
# DATAOPS group - data operations
# ===========================================================================


def _pipe_file() -> str:
    return _store_path("pipelines.json")


def _pipes() -> Dict[str, Any]:
    return _load_store(_pipe_file(), {}) or {}


spec("dataops", "pipeline-add", "Add a data pipeline definition.",
     [("name", "etl", "Pipeline name"), ("stages", "extract,transform,load", "Comma-separated stages")],
     lambda p: _ok(**_save_store(_pipe_file(), {**_pipes(), p["name"]: {"stages": [s.strip() for s in p["stages"].split(",")], "created": _now_iso()}})))

spec("dataops", "pipeline-list", "List data pipelines.",
     [],
     lambda p: _ok(pipelines=[{"name": k, "stages": v.get("stages", [])} for k, v in _pipes().items()]))

spec("dataops", "pipeline-run", "Simulate a pipeline run.",
     [("name", "etl", "Pipeline name")],
     lambda p: (_ok(pipeline=p["name"], status="completed",
                    stages=[{"stage": s, "status": "ok"} for s in (_pipes().get(p["name"], {}).get("stages") or ["extract", "transform", "load"])])
                if p["name"] in _pipes() else _err("pipeline not found")))

spec("dataops", "job-add", "Add a data job.",
     [("name", "nightly", "Job name"), ("cron", "0 2 * * *", "Cron expression"), ("action", "pipeline-run etl", "Action")],
     lambda p: _ok(**_save_store(_store_path("jobs.json"), {**(_load_store(_store_path("jobs.json"), {}) or {}), p["name"]: {"cron": p["cron"], "action": p["action"]}})))

spec("dataops", "job-list", "List data jobs.",
     [],
     lambda p: _ok(jobs=[{"name": k, "cron": v.get("cron")} for k, v in (_load_store(_store_path("jobs.json"), {}) or {}).items()]))

spec("dataops", "lineage", "Build a simple lineage graph.",
     [("source", "raw.db", "Source"), ("transform", "clean.py", "Transform"), ("sink", "warehouse.db", "Sink")],
     lambda p: _ok(lineage={"nodes": [p["source"], p["transform"], p["sink"]], "edges": [[p["source"], p["transform"]], [p["transform"], p["sink"]]]}))


# ===========================================================================
# CRYPTO2 group - more crypto helpers
# ===========================================================================


def _rot13(text: str) -> str:
    return text.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"))


def _caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    return "".join(out)


spec("crypto2", "base64-encode", "Encode text to base64.",
     [("text", "hello world", "Text")],
     lambda p: _ok(encoded=base64.b64encode(p["text"].encode("utf-8")).decode("ascii")))

spec("crypto2", "base64-decode", "Decode base64 to text.",
     [("text", "aGVsbG8gd29ybGQ=", "Base64 text")],
     lambda p: _ok(decoded=base64.b64decode(p["text"]).decode("utf-8", errors="replace")))

spec("crypto2", "rot13", "Apply ROT13 cipher.",
     [("text", "hello", "Text")],
     lambda p: _ok(result=_rot13(p["text"])))

spec("crypto2", "caesar", "Apply Caesar cipher with shift.",
     [("text", "hello", "Text"), ("shift", 3, "Shift")],
     lambda p: _ok(result=_caesar(p["text"], int(p["shift"]))))

spec("crypto2", "url-encode", "URL-encode a string.",
     [("text", "a b&c=d", "Text")],
     lambda p: _ok(encoded=urllib.parse.quote(p["text"], safe="")))

spec("crypto2", "url-decode", "URL-decode a string.",
     [("text", "a%20b%26c%3Dd", "Encoded text")],
     lambda p: _ok(decoded=urllib.parse.unquote(p["text"])))

spec("crypto2", "uuid", "Generate UUIDs.",
     [("n", 1, "Count"), ("version", 4, "4|1")],
     lambda p: _ok(uuids=[str(uuid.uuid4()) if p["version"] == "4" else str(uuid.uuid1()) for _ in range(int(p["n"]))]))

spec("crypto2", "random-hex", "Generate random hex string.",
     [("bytes", 16, "Byte count")],
     lambda p: _ok(hex=os.urandom(int(p["bytes"])).hex()))

spec("crypto2", "xor", "XOR two hex strings.",
     [("a", "aabb", "Hex A"), ("b", "00ff", "Hex B")],
     lambda p: _ok(result=bytes(x ^ y for x, y in zip(bytes.fromhex(p["a"]), bytes.fromhex(p["b"]))).hex()))

spec("crypto2", "checksum-text", "Checksum of text (adler32).",
     [("text", "hello", "Text")],
     lambda p: _ok(adler32=zlib.adler32(p["text"].encode("utf-8")), crc32=zlib.crc32(p["text"].encode("utf-8"))))

spec("crypto2", "compress", "Compress text (zlib) to hex.",
     [("text", "hello hello hello hello", "Text")],
     lambda p: _ok(compressed=zlib.compress(p["text"].encode("utf-8")).hex(), original_len=len(p["text"])))

spec("crypto2", "decompress", "Decompress zlib hex to text.",
     [("hex", "", "Compressed hex")],
     lambda p: _ok(decompressed=zlib.decompress(bytes.fromhex(p["hex"])).decode("utf-8", errors="replace")) if p["hex"] else _err("no hex input"))


# ===========================================================================
# GEO group - geo helpers
# ===========================================================================


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


spec("geo", "distance", "Haversine distance (km) between two coordinates.",
     [("lat1", 40.1776, "Lat 1"), ("lon1", 44.5126, "Lon 1"), ("lat2", 40.7306, "Lat 2"), ("lon2", -73.9352, "Lon 2")],
     lambda p: _ok(distance_km=round(_haversine(p["lat1"], p["lon1"], p["lat2"], p["lon2"]), 2)))

spec("geo", "bbox", "Compute bounding box around a point.",
     [("lat", 40.1776, "Latitude"), ("lon", 44.5126, "Longitude"), ("radius_km", 10.0, "Radius km")],
     lambda p: _ok(bbox={"min_lat": round(p["lat"] - p["radius_km"] / 111.0, 4),
                         "max_lat": round(p["lat"] + p["radius_km"] / 111.0, 4),
                         "min_lon": round(p["lon"] - p["radius_km"] / (111.0 * math.cos(math.radians(p["lat"])) or 1), 4),
                         "max_lon": round(p["lon"] + p["radius_km"] / (111.0 * math.cos(math.radians(p["lat"])) or 1), 4)}))

spec("geo", "dms", "Convert decimal degrees to DMS.",
     [("deg", 40.1776, "Decimal degrees")],
     lambda p: _ok(dms={"degrees": int(p["deg"]), "minutes": int((abs(p["deg"]) % 1) * 60),
                        "seconds": round(((abs(p["deg"]) % 1) * 60 % 1) * 60, 2)}))

spec("geo", "center", "Center of a list of coordinates (lat,lon pairs).",
     [("points", "40.1,44.5;40.2,44.6", "Semicolon-separated lat,lon pairs")],
     lambda p: _ok(center={"lat": round(statistics.mean([float(pt.split(",")[0]) for pt in p["points"].split(";")]), 4),
                           "lon": round(statistics.mean([float(pt.split(",")[1]) for pt in p["points"].split(";")]), 4)}))


# ===========================================================================
# STATS2 group - more statistics
# ===========================================================================


def _entropy(vals: List[float]) -> float:
    total = sum(vals) or 1
    return -sum((v / total) * math.log2(v / total) for v in vals if v > 0)


def _t_test(a: List[float], b: List[float]) -> Dict[str, Any]:
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b)) or 1
    return {"t_stat": round((ma - mb) / se, 4), "df": len(a) + len(b) - 2}


spec("stats2", "entropy", "Shannon entropy of a probability vector.",
     [("probs", "0.5,0.5", "Comma-separated probabilities")],
     lambda p: _ok(entropy=round(_entropy(_floats(p["probs"])), 4)))

spec("stats2", "t-test", "Welch t-test between two samples.",
     [("a", "1,2,3,4,5", "Sample A"), ("b", "2,3,4,5,6", "Sample B")],
     lambda p: _ok(**_t_test(_floats(p["a"]), _floats(p["b"]))))

spec("stats2", "chi2", "Chi-square statistic for observed vs expected.",
     [("observed", "10,20,30", "Observed counts"), ("expected", "15,20,25", "Expected counts")],
     lambda p: _ok(chi2=round(sum((o - e) ** 2 / (e or 1) for o, e in zip(_floats(p["observed"]), _floats(p["expected"]))), 4)))

spec("stats2", "mutual-info", "Normalized mutual information of two label lists.",
     [("a", "a,a,b,b,c", "Labels A"), ("b", "a,b,a,b,c", "Labels B")],
     lambda p: _ok(nmi=round((lambda la, lb: len(set(la) & set(lb)) / (len(set(la) | set(lb)) or 1))(
         p["a"].split(","), p["b"].split(",")), 4)))

spec("stats2", "percentile", "Compute a percentile of a list.",
     [("values", "1,2,3,4,5,6,7,8,9,10", "Comma-separated numbers"), ("p", 90.0, "Percentile 0-100")],
     lambda p: _ok(percentile=round(sorted(_floats(p["values"]))[min(len(_floats(p["values"])) - 1, int(p["p"] * len(_floats(p["values"])) / 100))], 4)))

spec("stats2", "range", "Range and IQR of a list.",
     [("values", "1,2,3,4,5,6,7,8,9,10", "Comma-separated numbers")],
     lambda p: _ok(range=max(_floats(p["values"])) - min(_floats(p["values"])),
                   iqr=round(sorted(_floats(p["values"]))[int(0.75 * (len(_floats(p["values"])) - 1))] - sorted(_floats(p["values"]))[int(0.25 * (len(_floats(p["values"])) - 1))], 4)))

spec("stats2", "zscore", "Z-score of a value in a distribution.",
     [("x", 6.0, "Value"), ("mean", 5.0, "Mean"), ("stdev", 2.0, "Std dev")],
     lambda p: _ok(zscore=round((p["x"] - p["mean"]) / (p["stdev"] or 1), 4)))

spec("stats2", "confidence", "Approximate 95% confidence interval of a mean.",
     [("values", "10,11,12,13,14", "Comma-separated numbers")],
     lambda p: _ok(ci95={"lo": round(statistics.mean(_floats(p["values"])) - 1.96 * (statistics.stdev(_floats(p["values"])) / math.sqrt(len(_floats(p["values"])))), 4),
                         "hi": round(statistics.mean(_floats(p["values"])) + 1.96 * (statistics.stdev(_floats(p["values"])) / math.sqrt(len(_floats(p["values"])))), 4)}))


# ===========================================================================
# MCP2 group - MCP registry helpers
# ===========================================================================


def _mcp_file() -> str:
    return _store_path("mcp_servers.json")


def _mcps() -> Dict[str, Any]:
    return _load_store(_mcp_file(), {}) or {}


spec("mcp2", "list", "List registered MCP servers.",
     [],
     lambda p: _ok(servers=[{"name": k, "url": v.get("url", ""), "tools": v.get("tools", 0)} for k, v in _mcps().items()]))

spec("mcp2", "add", "Register an MCP server.",
     [("name", "github", "Server name"), ("url", "http://localhost:9000", "Server URL"), ("tools", 10, "Tool count")],
     lambda p: _ok(**_save_store(_mcp_file(), {**_mcps(), p["name"]: {"url": p["url"], "tools": int(p["tools"])}})))

spec("mcp2", "remove", "Remove an MCP server registration.",
     [("name", "github", "Server name")],
     lambda p: _ok(removed=p["name"], **(_save_store(_mcp_file(), {k: v for k, v in _mcps().items() if k != p["name"]}))) if p["name"] in _mcps() else _err("not found"))

spec("mcp2", "check", "Check MCP server reachability.",
     [("name", "github", "Server name")],
     lambda p: _ok(name=p["name"], reachable=_port_open(
         urllib.parse.urlparse(_mcps().get(p["name"], {}).get("url", "http://localhost:9000")).hostname or "localhost",
         urllib.parse.urlparse(_mcps().get(p["name"], {}).get("url", "http://localhost:9000")).port or 9000)))


# ===========================================================================
# AGENT2 group - more agent orchestration
# ===========================================================================


def _runs_file() -> str:
    return _store_path("agent_runs.json")


def _runs() -> Dict[str, Any]:
    return _load_store(_runs_file(), {}) or {}


spec("agent2", "spawn", "Spawn a new agent run.",
     [("agent", "assistant", "Agent name"), ("task", "analyze data", "Task")],
     lambda p: _ok(run_id=str(uuid.uuid4())[:8], agent=p["agent"], task=p["task"], status="running",
                   **_save_store(_runs_file(), {**_runs(), str(uuid.uuid4())[:8]: {"agent": p["agent"], "task": p["task"], "status": "running", "started": _now_iso()}})))

spec("agent2", "list-runs", "List agent runs.",
     [],
     lambda p: _ok(runs=[{"id": k, "agent": v.get("agent"), "status": v.get("status")} for k, v in _runs().items()]))

spec("agent2", "stop", "Stop an agent run.",
     [("run_id", "abc123", "Run ID")],
     lambda p: _ok(run_id=p["run_id"], status="stopped") if p["run_id"] in _runs() else _err("run not found"))

spec("agent2", "status", "Show agent run status.",
     [("run_id", "abc123", "Run ID")],
     lambda p: _ok(**_runs().get(p["run_id"], {})) if p["run_id"] in _runs() else _err("run not found"))

spec("agent2", "scale", "Scale agent replicas (simulated).",
     [("agent", "assistant", "Agent name"), ("replicas", 3, "Replica count")],
     lambda p: _ok(agent=p["agent"], replicas=int(p["replicas"]), status="scaled"))

spec("agent2", "heartbeat", "Record agent heartbeat.",
     [("agent", "assistant", "Agent name")],
     lambda p: _ok(agent=p["agent"], ts=_now_iso(), status="alive"))


# ===========================================================================
# WORKFLOW2 group - more workflow
# ===========================================================================


def _wf_file2() -> str:
    return _store_path("workflows2.json")


def _wf2s() -> Dict[str, Any]:
    return _load_store(_wf_file2(), {}) or {}


spec("workflow2", "trigger", "Trigger a workflow with input.",
     [("name", "build", "Workflow name"), ("input", "{}", "JSON input")],
     lambda p: _ok(workflow=p["name"], input=p["input"], status="queued", queued_at=_now_iso()))

spec("workflow2", "retry", "Retry a failed workflow step.",
     [("name", "build", "Workflow name"), ("step", "compile", "Step name"), ("attempts", 3, "Max attempts")],
     lambda p: _ok(workflow=p["name"], step=p["step"], attempts=int(p["attempts"]), status="retrying"))

spec("workflow2", "status", "Check workflow status.",
     [("name", "build", "Workflow name")],
     lambda p: _ok(workflow=p["name"], status="completed" if p["name"] in _wf2s() else "not_found",
                   last_run=_wf2s().get(p["name"], {}).get("last_run")))

spec("workflow2", "add", "Add a workflow definition.",
     [("name", "build", "Workflow name"), ("steps", "checkout,build,test", "Comma-separated steps")],
     lambda p: _ok(**_save_store(_wf_file2(), {**_wf2s(), p["name"]: {"steps": [s.strip() for s in p["steps"].split(",")], "last_run": _now_iso()}})))

spec("workflow2", "list", "List workflow definitions.",
     [],
     lambda p: _ok(workflows=[{"name": k, "steps": v.get("steps", [])} for k, v in _wf2s().items()]))


# ===========================================================================
# MONITOR2 group - more monitoring
# ===========================================================================


def _loadavg() -> List[float]:
    try:
        with open("/proc/loadavg") as f:
            return [float(x) for x in f.read().split()[:3]]
    except Exception:
        return []


spec("monitor2", "cpu", "CPU load averages.",
     [],
     lambda p: _ok(loadavg=_loadavg(), cpu_count=os.cpu_count()))

spec("monitor2", "mem", "Memory info (if available).",
     [],
     lambda p: _ok(memory=_run("free -m 2>/dev/null || echo unavailable").splitlines()))

spec("monitor2", "disk", "Disk usage for root.",
     [],
     lambda p: _ok(root_usage=_run("df -h / 2>/dev/null || echo unavailable").splitlines()))

spec("monitor2", "net", "Network interface summary.",
     [],
     lambda p: _ok(interfaces=_run("ip -brief addr 2>/dev/null || ifconfig 2>/dev/null || echo unavailable").splitlines()[:20]))

spec("monitor2", "uptime", "System uptime.",
     [],
     lambda p: _ok(uptime=_run("uptime 2>/dev/null || echo unavailable")))

spec("monitor2", "process-count", "Number of running processes.",
     [],
     lambda p: _ok(count=int(_run("ps -e --no-headers 2>/dev/null | wc -l") or "0")))

spec("monitor2", "alerts", "Generate sample alert rules.",
     [],
     lambda p: _ok(alerts=[{"metric": "cpu", "threshold": 0.9, "action": "scale"},
                           {"metric": "mem", "threshold": 0.85, "action": "alert"},
                           {"metric": "error_rate", "threshold": 0.05, "action": "rollback"}]))

spec("monitor2", "synthetic", "Generate synthetic health check results.",
     [("checks", "api,db,cache", "Comma-separated check names")],
     lambda p: _ok(results=[{"check": c.strip(), "status": "pass" if random.random() > 0.1 else "warn", "latency_ms": round(random.uniform(5, 200), 1)}
                            for c in p["checks"].split(",") if c.strip()]))


# ===========================================================================
# Register with the main bulk registry
# ===========================================================================
_bulk.rebuild_index()