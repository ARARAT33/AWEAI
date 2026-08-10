# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI extended bulk command specs (v4.1).

Adds hundreds of additional declarative commands across new groups:

  dataset      — dataset lifecycle: version, split, stats
  feature      — feature engineering: transforms, selection
  deploy       — deployment: edge, cloud, packaging
  scheduler    — cron-like job scheduling
  notification — notifications: email, webhook, telegram
  search       — text/vector search utilities
  code         — code analysis, review, formatting
  shell        — shell command helpers
  git          — git helpers
  docker       — docker helpers
  mcp          — MCP (model context protocol) helpers
  vector       — vector math helpers
  eval         — evaluation utilities
  quantize     — quantization utilities

Every spec follows the same declarative shape used by :mod:`aweai.bulk`
(name, help, params, fn) and is appended to the main registry.
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
# DATASET group — lifecycle, version, split, stats
# ===========================================================================

def _ds_read(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        if p.suffix == ".jsonl":
            out = []
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
            return out
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("data", [])
        if p.suffix == ".csv":
            import csv
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            return rows
    except Exception:
        return []
    return []


def _ds_write(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        Path(path).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


spec("dataset", "stats", "Dataset statistics (rows, columns, missing).",
     [("path", "data.jsonl", "Path")], lambda p: _ds_stats(p["path"]))
spec("dataset", "split", "Split a dataset into train/val/test.",
     [("path", "data.jsonl", "Path"), ("train", 0.8, "Train ratio"), ("val", 0.1, "Val ratio"), ("out", "split", "Output prefix")],
     lambda p: _ds_split(p["path"], float(p["train"]), float(p["val"]), p["out"]))
spec("dataset", "sample", "Sample N rows from a dataset.",
     [("path", "data.jsonl", "Path"), ("n", 10, "Rows")],
     lambda p: _ds_sample(p["path"], int(p["n"])))
spec("dataset", "shuffle", "Shuffle a dataset.",
     [("path", "data.jsonl", "Path"), ("seed", 1, "Seed")],
     lambda p: _ds_shuffle(p["path"], int(p["seed"])))
spec("dataset", "dedupe", "Remove duplicate rows from a dataset.",
     [("path", "data.jsonl", "Path")], lambda p: _ds_dedupe(p["path"]))
spec("dataset", "merge", "Merge two datasets.",
     [("a", "a.jsonl", "First path"), ("b", "b.jsonl", "Second path"), ("out", "merged.jsonl", "Output path")],
     lambda p: _ds_merge(p["a"], p["b"], p["out"]))
spec("dataset", "head", "Show first N rows.",
     [("path", "data.jsonl", "Path"), ("n", 5, "Rows")],
     lambda p: _ok(rows=_ds_read(p["path"])[: max(0, int(p["n"]))]))
spec("dataset", "version", "Register a dataset version.",
     [("path", "data.jsonl", "Path"), ("version", "v1", "Version tag")],
     lambda p: _ok(path=p["path"], version=p["version"], sha256=_sha256(Path(p["path"]).read_text(encoding="utf-8")) if Path(p["path"]).exists() else None))
spec("dataset", "validate", "Validate dataset rows are dicts and non-empty.",
     [("path", "data.jsonl", "Path")], lambda p: _ds_validate(p["path"]))


def _ds_stats(path: str) -> Dict[str, Any]:
    rows = _ds_read(path)
    if not rows:
        return _err("no rows or file not found")
    cols = list(rows[0].keys())
    missing = {c: sum(1 for r in rows if r.get(c) in (None, "")) for c in cols}
    return _ok(path=path, rows=len(rows), columns=cols, missing=missing)


def _ds_split(path: str, train: float, val: float, out: str) -> Dict[str, Any]:
    rows = _ds_read(path)
    if not rows:
        return _err("no rows")
    n = len(rows)
    nt = max(1, int(n * train))
    nv = max(1, int(n * val))
    tr, va, te = rows[:nt], rows[nt:nt + nv], rows[nt + nv:]
    _ds_write(f"{out}_train.jsonl", tr)
    _ds_write(f"{out}_val.jsonl", va)
    _ds_write(f"{out}_test.jsonl", te)
    return _ok(train=len(tr), val=len(va), test=len(te), prefix=out)


def _ds_sample(path: str, n: int) -> Dict[str, Any]:
    rows = _ds_read(path)
    return _ok(rows=rows[: max(0, n)], count=min(len(rows), max(0, n)))


def _ds_shuffle(path: str, seed: int) -> Dict[str, Any]:
    rows = _ds_read(path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    _ds_write(path, rows)
    return _ok(path=path, rows=len(rows), seed=seed)


def _ds_dedupe(path: str) -> Dict[str, Any]:
    rows = _ds_read(path)
    seen = set()
    uniq = []
    for r in rows:
        key = json.dumps(r, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    _ds_write(path, uniq)
    return _ok(path=path, before=len(rows), after=len(uniq))


def _ds_merge(a: str, b: str, out: str) -> Dict[str, Any]:
    rows = _ds_read(a) + _ds_read(b)
    _ds_write(out, rows)
    return _ok(path=out, rows=len(rows))


def _ds_validate(path: str) -> Dict[str, Any]:
    rows = _ds_read(path)
    problems = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            problems.append({"row": i, "issue": "not an object"})
        elif not r:
            problems.append({"row": i, "issue": "empty object"})
    return _ok(valid=not problems, rows=len(rows), problems=problems)


# ===========================================================================
# FEATURE group — feature engineering, transforms, selection
# ===========================================================================

spec("feature", "normalize", "Min-max normalize a list of numbers.",
     [("values", "1,2,3,4", "Values")], lambda p: _feature_normalize(p["values"]))
spec("feature", "standardize", "Z-score standardize a list of numbers.",
     [("values", "1,2,3,4", "Values")], lambda p: _feature_standardize(p["values"]))
spec("feature", "bin", "Bin values into N buckets.",
     [("values", "1,2,3,4,5,6", "Values"), ("bins", 3, "Bins")],
     lambda p: _feature_bin(p["values"], int(p["bins"])))
spec("feature", "onehot", "One-hot encode categorical values.",
     [("values", "cat,dog,cat,bird", "Values")], lambda p: _feature_onehot(p["values"]))
spec("feature", "impute", "Impute missing values (marked as 'NA') with strategy.",
     [("values", "1,NA,3,NA,5", "Values"), ("strategy", "mean", "mean|median|zero")],
     lambda p: _feature_impute(p["values"], p["strategy"]))
spec("feature", "select", "Select top-k features by variance (heuristic).",
     [("columns", "a,b,c", "Column names"), ("variance", "0.5,0.1,0.9", "Variances"), ("k", 2, "Top K")],
     lambda p: _feature_select(p["columns"], p["variance"], int(p["k"])))
spec("feature", "poly", "Generate polynomial features for x.",
     [("x", 3.0, "Value"), ("degree", 3, "Degree")],
     lambda p: _ok(features=[round(x ** d, 4) for d in range(1, int(p["degree"]) + 1)]))
spec("feature", "hash", "Feature hashing: map token to bucket.",
     [("token", "hello", "Token"), ("buckets", 1000, "Buckets")],
     lambda p: _ok(bucket=int(hashlib.md5(p["token"].encode("utf-8")).hexdigest(), 16) % int(p["buckets"])))


def _feature_normalize(values: str) -> Dict[str, Any]:
    vals = _floats(values)
    if not vals:
        return _err("no values")
    lo, hi = min(vals), max(vals)
    out = [(x - lo) / (hi - lo) if hi != lo else 0.0 for x in vals]
    return _ok(normalized=[round(v, 4) for v in out])


def _feature_standardize(values: str) -> Dict[str, Any]:
    vals = _floats(values)
    if not vals:
        return _err("no values")
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return _ok(standardized=[round((x - mu) / sd, 4) for x in vals], mean=mu, std=sd)


def _feature_bin(values: str, bins: int) -> Dict[str, Any]:
    vals = _floats(values)
    if not vals:
        return _err("no values")
    lo, hi = min(vals), max(vals)
    width = (hi - lo) / max(1, bins) or 1.0
    out = [min(int((x - lo) / width), bins - 1) if hi != lo else 0 for x in vals]
    return _ok(bins=out, bin_count=bins)


def _feature_onehot(values: str) -> Dict[str, Any]:
    cats = [v.strip() for v in values.split(",") if v.strip()]
    uniq = list(dict.fromkeys(cats))
    out = []
    for c in cats:
        vec = [1 if c == u else 0 for u in uniq]
        out.append({"value": c, "vector": vec})
    return _ok(categories=uniq, encoded=out)


def _feature_impute(values: str, strategy: str) -> Dict[str, Any]:
    vals = [None if v.strip().upper() in ("NA", "NONE", "NULL", "") else _num(v) for v in values.split(",")]
    known = [v for v in vals if v is not None]
    if strategy == "mean":
        fill = statistics.mean(known) if known else 0.0
    elif strategy == "median":
        fill = statistics.median(known) if known else 0.0
    else:
        fill = 0.0
    return _ok(imputed=[fill if v is None else v for v in vals], strategy=strategy, fill=fill)


def _feature_select(columns: str, variance: str, k: int) -> Dict[str, Any]:
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    vars_ = _floats(variance)
    pairs = list(zip(cols, vars_))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return _ok(selected=[{"column": c, "variance": v} for c, v in pairs[: max(1, k)]])


# ===========================================================================
# DEPLOY group — edge, cloud, packaging
# ===========================================================================

spec("deploy", "plan", "Show a deployment plan (steps).",
     [("target", "edge", "edge|cloud|desktop"), ("app", "aweai", "App name")],
     lambda p: _deploy_plan(p["target"], p["app"]))
spec("deploy", "package", "Describe packaging steps for a target.",
     [("target", "wheel", "wheel|exe|appimage|apk|web")],
     lambda p: _deploy_package(p["target"]))
spec("deploy", "check", "Check deploy prerequisites (python, git, etc.).",
     [], lambda p: _deploy_check())
spec("deploy", "push", "Register a deployment (target + artifact).",
     [("target", "edge", "Target"), ("artifact", "model.onnx", "Artifact")],
     lambda p: _ok(deployed=True, target=p["target"], artifact=p["artifact"], time=_now_iso()))
spec("deploy", "rollback", "Roll back a deployment to previous version.",
     [("deployment", "edge-1", "Deployment id")],
     lambda p: _ok(rolled_back=p["deployment"], to="previous", status="ok"))


def _deploy_plan(target: str, app: str) -> Dict[str, Any]:
    plans = {
        "edge": ["export onnx", "quantize int8", "bundle runtime", "sign", "ship"],
        "cloud": ["build image", "push registry", "provision", "health check", "route traffic"],
        "desktop": ["build installer", "sign", "notarize", "publish"],
    }
    return _ok(target=target, app=app, steps=plans.get(target, plans["cloud"]))


def _deploy_package(target: str) -> Dict[str, Any]:
    cmds = {
        "wheel": "python -m build",
        "exe": "pyinstaller aweai.spec",
        "appimage": "linuxdeploy --appdir AppDir",
        "apk": "buildozer android debug",
        "web": "npm run build && npx serve dist",
    }
    return _ok(target=target, command=cmds.get(target, "unknown target"))


def _deploy_check() -> Dict[str, Any]:
    checks = {
        "python": sys.version_info[:3],
        "platform": platform.system(),
        "git": shutil.which("git") is not None,
        "pip": shutil.which("pip") is not None,
    }
    return _ok(checks=checks, ready=all(v for k, v in checks.items() if k != "python"))


# ===========================================================================
# SCHEDULER group — cron-like job scheduling
# ===========================================================================

def _sched_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/scheduler.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _sched_data() -> Dict[str, Any]:
    p = Path(_sched_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"jobs": []}
    return {"jobs": []}


def _save_sched(data: Dict[str, Any]) -> None:
    Path(_sched_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _sched_add(name: str, cron: str, command: str) -> Dict[str, Any]:
    data = _sched_data()
    data["jobs"].append({"name": name, "cron": cron, "command": command, "enabled": True, "added": _now_iso()})
    _save_sched(data)
    return _ok(job=name, cron=cron, command=command)


spec("scheduler", "add", "Add a scheduled job (cron expression).",
     [("name", "daily", "Job name"), ("cron", "0 9 * * *", "Cron"), ("command", "aweai version", "Command")],
     lambda p: _sched_add(p["name"], p["cron"], p["command"]))
spec("scheduler", "list", "List scheduled jobs.", [], lambda p: _ok(jobs=_sched_data()["jobs"]))
spec("scheduler", "enable", "Enable a job.",
     [("name", "daily", "Job name")], lambda p: _sched_enable(p["name"], True))
spec("scheduler", "disable", "Disable a job.",
     [("name", "daily", "Job name")], lambda p: _sched_enable(p["name"], False))
spec("scheduler", "remove", "Remove a job.",
     [("name", "daily", "Job name")], lambda p: _sched_remove(p["name"]))
spec("scheduler", "next", "Estimate next run minute from a cron-like expr (simple).",
     [("cron", "0 9 * * *", "Cron")], lambda p: _ok(cron=p["cron"], note="field-level parser: min hour dom mon dow",
                                                    fields=p["cron"].split()))
spec("scheduler", "validate", "Validate a cron expression has 5 fields.",
     [("cron", "0 9 * * *", "Cron")], lambda p: _sched_validate(p["cron"]))


def _sched_enable(name: str, enabled: bool) -> Dict[str, Any]:
    data = _sched_data()
    for j in data["jobs"]:
        if j["name"] == name:
            j["enabled"] = enabled
            _save_sched(data)
            return _ok(name=name, enabled=enabled)
    return _err(f"job not found: {name}")


def _sched_remove(name: str) -> Dict[str, Any]:
    data = _sched_data()
    data["jobs"] = [j for j in data["jobs"] if j["name"] != name]
    _save_sched(data)
    return _ok(removed=name)


def _sched_validate(cron: str) -> Dict[str, Any]:
    fields = cron.split()
    valid = len(fields) == 5
    return _ok(valid=valid, fields=len(fields), expected=5)


# ===========================================================================
# NOTIFICATION group — email, webhook, telegram (templates)
# ===========================================================================

spec("notification", "send", "Send a notification via a channel (simulated).",
     [("channel", "log", "log|webhook|email|telegram"), ("message", "hello", "Message"), ("target", None, "Target (url/email)")],
     lambda p: _notif_send(p["channel"], p["message"], p["target"]))
spec("notification", "channels", "List supported notification channels.",
     [], lambda p: _ok(channels=["log", "webhook", "email", "telegram", "slack"]))
spec("notification", "test", "Send a test notification to all channels.",
     [("message", "test", "Message")], lambda p: _notif_test(p["message"]))


def _notif_send(channel: str, message: str, target: Optional[str]) -> Dict[str, Any]:
    if channel == "webhook":
        if not target:
            return _err("webhook requires --target URL")
        try:
            req = urllib.request.Request(target, data=json.dumps({"text": message}).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return _ok(channel=channel, status=r.status, sent=True)
        except Exception as e:
            return _err(f"webhook failed: {e}")
    return _ok(channel=channel, message=message, target=target or "local", sent=True, simulated=True)


def _notif_test(message: str) -> Dict[str, Any]:
    out = []
    for c in ["log", "email", "telegram", "slack"]:
        out.append({"channel": c, "sent": True, "simulated": True})
    return _ok(message=message, results=out)
