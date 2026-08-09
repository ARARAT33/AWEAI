"""AWEAI monitoring tools — metrics, alerts, health checks, tracing.

Each tool has a unique purpose and works with the standard library.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.config import ensure_runtime_dirs
from aweai.tools.registry import tool


def _metrics_path() -> Path:
    return ensure_runtime_dirs()["data"] / "aweai_metrics.json"


def _load_metrics() -> Dict[str, Any]:
    p = _metrics_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_metrics(m: Dict[str, Any]) -> None:
    _metrics_path().write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


@tool("metric_record", "monitoring", "Record a named metric value (with timestamp)")
def metric_record(name: str, value: float) -> Dict[str, Any]:
    metrics = _load_metrics()
    if name not in metrics:
        metrics[name] = []
    metrics[name].append({"value": value, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    metrics[name] = metrics[name][-1000:]
    _save_metrics(metrics)
    return {"name": name, "recorded": value, "samples": len(metrics[name])}


@tool("metric_get", "monitoring", "Read recorded values of a metric")
def metric_get(name: str) -> Dict[str, Any]:
    metrics = _load_metrics()
    return {"name": name, "values": metrics.get(name, []), "count": len(metrics.get(name, []))}


@tool("metric_list", "monitoring", "List all metric names and sample counts")
def metric_list() -> Dict[str, Any]:
    metrics = _load_metrics()
    return {"metrics": {k: len(v) for k, v in metrics.items()}, "count": len(metrics)}


@tool("metric_summary", "monitoring", "Summary statistics of a metric (min, max, mean, last)")
def metric_summary(name: str) -> Dict[str, Any]:
    metrics = _load_metrics()
    values = [s["value"] for s in metrics.get(name, [])]
    if not values:
        return {"name": name, "error": "no samples"}
    return {
        "name": name,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "last": values[-1],
    }


@tool("metric_clear", "monitoring", "Clear all recorded metrics")
def metric_clear() -> Dict[str, Any]:
    _save_metrics({})
    return {"cleared": True}


@tool("health_check", "monitoring", "Run a health check on a URL and report latency/status")
def health_check(url: str = "http://127.0.0.1:8888/api/health", timeout: int = 10) -> Dict[str, Any]:
    import urllib.request

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            latency = round((time.time() - start) * 1000, 2)
            return {"url": url, "status": r.status, "latency_ms": latency, "healthy": r.status < 400}
    except Exception as e:
        latency = round((time.time() - start) * 1000, 2)
        return {"url": url, "error": str(e), "latency_ms": latency, "healthy": False}


@tool("monitor_snapshot", "monitoring", "Take a one-shot system snapshot (cpu, memory, load, disk)")
def monitor_snapshot() -> Dict[str, Any]:
    import os

    snap = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        snap["cpu_count"] = os.cpu_count()
        snap["load"] = list(os.getloadavg())
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal") or line.startswith("MemAvailable"):
                    parts = line.split(":")
                    snap[parts[0].strip()] = parts[1].strip()
    except Exception:
        pass
    return {"snapshot": snap}


@tool("uptime_monitor", "monitoring", "System uptime + process start time of the current PID")
def uptime_monitor() -> Dict[str, Any]:
    import os

    out = {"pid": os.getpid()}
    try:
        with open("/proc/uptime", encoding="utf-8") as f:
            out["system_uptime_seconds"] = float(f.read().split()[0])
    except Exception:
        pass
    try:
        with open(f"/proc/{os.getpid()}/stat", encoding="utf-8") as f:
            parts = f.read().split()
            out["process_start_ticks"] = parts[21]
    except Exception:
        pass
    return out


@tool("trace_start", "monitoring", "Start a named trace (records start time)")
def trace_start(name: str = "trace1") -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_traces.json"
    traces = {}
    if path.exists():
        try:
            traces = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            traces = {}
    traces[name] = {"start": time.time(), "end": None, "duration_ms": None}
    path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    return {"name": name, "started": True}


@tool("trace_end", "monitoring", "End a named trace and report duration")
def trace_end(name: str = "trace1") -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_traces.json"
    if not path.exists():
        return {"error": "no traces stored"}
    traces = json.loads(path.read_text(encoding="utf-8"))
    if name not in traces:
        return {"error": f"trace '{name}' not found"}
    start = traces[name].get("start")
    duration = round((time.time() - start) * 1000, 2) if start else None
    traces[name]["end"] = time.time()
    traces[name]["duration_ms"] = duration
    path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    return {"name": name, "duration_ms": duration}


@tool("trace_list", "monitoring", "List all traces and their durations")
def trace_list() -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_traces.json"
    if not path.exists():
        return {"traces": {}}
    return {"traces": json.loads(path.read_text(encoding="utf-8"))}


@tool("alert_check_threshold", "monitoring", "Check whether a metric exceeds a threshold")
def alert_check_threshold(name: str, threshold: float, op: str = ">") -> Dict[str, Any]:
    metrics = _load_metrics()
    values = [s["value"] for s in metrics.get(name, [])]
    if not values:
        return {"name": name, "error": "no samples"}
    last = values[-1]
    triggered = last > threshold if op == ">" else last < threshold if op == "<" else last == threshold
    return {"name": name, "last": last, "threshold": threshold, "op": op, "triggered": triggered}


@tool("log_append", "monitoring", "Append a line to a JSONL log file")
def log_append(message: str, level: str = "info", logger: str = "app") -> Dict[str, Any]:
    from aweai.config import ensure_runtime_dirs

    logs_dir = ensure_runtime_dirs()["logs"]
    path = logs_dir / f"{logger}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"level": level, "message": message, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    return {"logger": logger, "level": level, "appended": True}


@tool("log_tail", "monitoring", "Read the last N lines of a JSONL log")
def log_tail(logger: str = "app", n: int = 20) -> Dict[str, Any]:
    from aweai.config import ensure_runtime_dirs

    path = ensure_runtime_dirs()["logs"] / f"{logger}.jsonl"
    if not path.exists():
        return {"error": f"log '{logger}' not found"}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"logger": logger, "entries": lines[-n:], "count": len(lines)}


@tool("log_clear", "monitoring", "Clear a JSONL log file")
def log_clear(logger: str = "app") -> Dict[str, Any]:
    from aweai.config import ensure_runtime_dirs

    path = ensure_runtime_dirs()["logs"] / f"{logger}.jsonl"
    if path.exists():
        path.unlink()
        return {"logger": logger, "cleared": True}
    return {"logger": logger, "cleared": False}


__all__ = []
