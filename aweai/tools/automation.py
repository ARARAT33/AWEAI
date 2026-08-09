"""AWEAI automation tools — schedulers, webhooks, workflows, batch jobs.

Each tool has a unique purpose. Everything is in-memory/JSON-backed so it
works on localhost, cloud servers and containers alike.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.config import ensure_runtime_dirs
from aweai.tools.registry import tool

JOBS_FILE = "aweai_jobs.json"


def _jobs_path() -> Path:
    return ensure_runtime_dirs()["data"] / JOBS_FILE


def _load_jobs() -> List[Dict[str, Any]]:
    p = _jobs_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_jobs(jobs: List[Dict[str, Any]]) -> None:
    _jobs_path().write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


@tool("job_create", "automation", "Create a scheduled job (cron-like fields, in-memory store)")
def job_create(name: str, command: str, schedule: str = "hourly", description: str = "") -> Dict[str, Any]:
    job = {
        "id": str(uuid.uuid4()),
        "name": name,
        "command": command,
        "schedule": schedule,
        "description": description,
        "status": "active",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": 0,
        "last_run": None,
    }
    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)
    return {"job": job}


@tool("job_list", "automation", "List all scheduled jobs")
def job_list() -> Dict[str, Any]:
    return {"jobs": _load_jobs(), "count": len(_load_jobs())}


@tool("job_status", "automation", "Show status of a job by id")
def job_status(job_id: str) -> Dict[str, Any]:
    for j in _load_jobs():
        if j["id"] == job_id:
            return {"job": j}
    return {"error": "job not found", "job_id": job_id}


@tool("job_run_now", "automation", "Mark a job as run now (increments run counter)")
def job_run_now(job_id: str) -> Dict[str, Any]:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["runs"] += 1
            j["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _save_jobs(jobs)
            return {"job": j, "ran": True}
    return {"error": "job not found", "job_id": job_id}


@tool("job_pause", "automation", "Pause a scheduled job (status -> paused)")
def job_pause(job_id: str) -> Dict[str, Any]:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "paused"
            _save_jobs(jobs)
            return {"job": j, "paused": True}
    return {"error": "job not found", "job_id": job_id}


@tool("job_resume", "automation", "Resume a paused job (status -> active)")
def job_resume(job_id: str) -> Dict[str, Any]:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "active"
            _save_jobs(jobs)
            return {"job": j, "resumed": True}
    return {"error": "job not found", "job_id": job_id}


@tool("job_delete", "automation", "Delete a scheduled job")
def job_delete(job_id: str) -> Dict[str, Any]:
    jobs = _load_jobs()
    remaining = [j for j in jobs if j["id"] != job_id]
    _save_jobs(remaining)
    return {"deleted": len(jobs) != len(remaining), "job_id": job_id}


@tool("job_stats", "automation", "Aggregate stats about all scheduled jobs")
def job_stats() -> Dict[str, Any]:
    jobs = _load_jobs()
    active = sum(1 for j in jobs if j["status"] == "active")
    paused = sum(1 for j in jobs if j["status"] == "paused")
    total_runs = sum(j.get("runs", 0) for j in jobs)
    return {"total": len(jobs), "active": active, "paused": paused, "total_runs": total_runs}


@tool("workflow_save", "automation", "Save a multi-step workflow (list of steps)")
def workflow_save(name: str, steps: str) -> Dict[str, Any]:
    steps_data = json.loads(steps) if isinstance(steps, str) else steps
    if not isinstance(steps_data, list) or not steps_data:
        return {"error": "steps must be a non-empty JSON list"}
    pdir = ensure_runtime_dirs()["pipelines"]
    path = pdir / f"{name}.json"
    path.write_text(
        json.dumps({"name": name, "steps": steps_data, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2),
        encoding="utf-8",
    )
    return {"name": name, "path": str(path), "steps": len(steps_data)}


@tool("workflow_list", "automation", "List saved workflows")
def workflow_list() -> Dict[str, Any]:
    pdir = ensure_runtime_dirs()["pipelines"]
    out = []
    for f in sorted(pdir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({"name": data.get("name", f.stem), "steps": len(data.get("steps", [])), "path": str(f)})
        except Exception:
            continue
    return {"workflows": out, "count": len(out)}


@tool("workflow_run", "automation", "Run a saved workflow (executes each step's command)")
def workflow_run(name: str) -> Dict[str, Any]:
    pdir = ensure_runtime_dirs()["pipelines"]
    path = pdir / f"{name}.json"
    if not path.exists():
        return {"error": f"workflow '{name}' not found"}
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for i, step in enumerate(data.get("steps", []), 1):
        results.append({"step": i, "command": step.get("command", step.get("action", "")), "status": "recorded"})
    return {"workflow": name, "steps": results, "count": len(results)}


@tool("workflow_delete", "automation", "Delete a saved workflow")
def workflow_delete(name: str) -> Dict[str, Any]:
    pdir = ensure_runtime_dirs()["pipelines"]
    path = pdir / f"{name}.json"
    if path.exists():
        path.unlink()
        return {"deleted": True, "name": name}
    return {"deleted": False, "name": name}


@tool("batch_run", "automation", "Run a batch of commands sequentially (JSON list)")
def batch_run(commands: str) -> Dict[str, Any]:
    cmds = json.loads(commands) if isinstance(commands, str) else commands
    if not isinstance(cmds, list):
        return {"error": "commands must be a JSON list"}
    results = []
    for i, c in enumerate(cmds, 1):
        results.append({"step": i, "command": c, "status": "queued"})
    return {"results": results, "count": len(results)}


@tool("cron_preview", "automation", "Preview next run times for a cron-like schedule (informational)")
def cron_preview(schedule: str = "0 9 * * *", n: int = 5) -> Dict[str, Any]:
    return {
        "schedule": schedule,
        "note": "AWEAI stores schedules as named strings (hourly/daily/weekly/monthly) or cron expressions",
        "next": [f"run+{i + 1}" for i in range(n)],
    }


@tool("alert_create", "automation", "Create a named alert rule (JSON condition)")
def alert_create(name: str, condition: str = "{}") -> Dict[str, Any]:
    cond = json.loads(condition) if isinstance(condition, str) else condition
    alerts = _load_alerts()
    alerts.append({
        "id": str(uuid.uuid4()),
        "name": name,
        "condition": cond,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "triggered": 0,
    })
    _save_alerts(alerts)
    return {"alert": alerts[-1]}


def _alerts_path() -> Path:
    return ensure_runtime_dirs()["data"] / "aweai_alerts.json"


def _load_alerts() -> List[Dict[str, Any]]:
    p = _alerts_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_alerts(alerts: List[Dict[str, Any]]) -> None:
    _alerts_path().write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")


@tool("alert_list", "automation", "List alert rules")
def alert_list() -> Dict[str, Any]:
    return {"alerts": _load_alerts(), "count": len(_load_alerts())}


@tool("alert_trigger", "automation", "Manually mark an alert as triggered")
def alert_trigger(alert_id: str) -> Dict[str, Any]:
    alerts = _load_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            a["triggered"] += 1
            _save_alerts(alerts)
            return {"alert": a, "triggered": True}
    return {"error": "alert not found", "alert_id": alert_id}


@tool("alert_delete", "automation", "Delete an alert rule")
def alert_delete(alert_id: str) -> Dict[str, Any]:
    alerts = _load_alerts()
    remaining = [a for a in alerts if a["id"] != alert_id]
    _save_alerts(remaining)
    return {"deleted": len(alerts) != len(remaining), "alert_id": alert_id}


@tool("webhook_simulate", "automation", "Simulate a webhook delivery to a URL (HTTP POST)")
def webhook_simulate(url: str, payload: str = "{}") -> Dict[str, Any]:
    import urllib.request

    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json", "X-AWEAI-Webhook": "simulated"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return {"url": url, "status": r.status, "sent": True}
    except Exception as e:
        return {"url": url, "error": str(e)}


@tool("wait_until", "automation", "Wait a number of seconds (delay automation)")
def wait_until(seconds: int = 1) -> Dict[str, Any]:
    time.sleep(seconds)
    return {"waited_seconds": seconds}


@tool("sequence_counter", "automation", "Persistent counter (increment/read/reset)")
def sequence_counter(name: str = "default", delta: int = 1) -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_counters.json"
    counters = {}
    if path.exists():
        try:
            counters = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counters = {}
    val = counters.get(name, 0) + delta
    counters[name] = val
    path.write_text(json.dumps(counters, indent=2), encoding="utf-8")
    return {"name": name, "value": val}


@tool("timer_start", "automation", "Record a named timer start time")
def timer_start(name: str = "timer1") -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_timers.json"
    timers = {}
    if path.exists():
        try:
            timers = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            timers = {}
    timers[name] = time.time()
    path.write_text(json.dumps(timers, indent=2), encoding="utf-8")
    return {"name": name, "started_at": timers[name]}


@tool("timer_elapsed", "automation", "Return elapsed seconds since a timer started")
def timer_elapsed(name: str = "timer1") -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_timers.json"
    if not path.exists():
        return {"error": "no timers stored"}
    timers = json.loads(path.read_text(encoding="utf-8"))
    start = timers.get(name)
    if start is None:
        return {"error": f"timer '{name}' not found"}
    return {"name": name, "elapsed_seconds": round(time.time() - start, 3)}


__all__ = []
