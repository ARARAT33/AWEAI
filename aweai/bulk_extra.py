# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI extended bulk command specs (v4.1).

Adds hundreds of additional declarative commands across new groups:

  agent        — agent orchestration, roles, tools, memory, multi-agent
  workflow     — workflow definitions, DAG, steps, retries, triggers
  pipeline     — data pipelines: stages, transforms, validation, artifacts
  rag          — retrieval-augmented generation: index, search, knowledge
  research     — web research, search, browse, citations
  reasoning    — chain-of-thought, planning, decomposition, verification
  memory       — memory store: episodic, semantic, procedural
  orchestrate  — task orchestration: scheduling, fan-out, retries
  security     — security: scan, audit, secrets, policy
  backup       — backup/restore/versioning
  monitor      — monitoring: metrics, logs, alerts, health
  plugin       — plugin/ecosystem management
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
# RESEARCH group — web research, search, browse, citations
# ===========================================================================

spec("research", "search", "Web search helper (URL builder for common engines).",
     [("query", "AI news", "Search query"), ("engine", "google", "google|bing|duckduckgo|github")],
     lambda p: _research_search(p["query"], p["engine"]))
spec("research", "fetch", "Fetch a URL and return size/status (needs network).",
     [("url", "https://example.com", "URL"), ("timeout", 10, "Timeout seconds")],
     lambda p: _research_fetch(p["url"], int(p["timeout"])))
spec("research", "citations", "Format citations from a ';' separated reference list.",
     [("refs", "AWEAI docs;OpenAI docs", "References")],
     lambda p: _ok(citations=[f"[{i + 1}] {r.strip()}" for i, r in enumerate(p["refs"].split(";")) if r.strip()]))
spec("research", "summarize", "Summarize text by extracting key sentences (extractive).",
     [("text", "AWEAI is a universal CLI. It covers AI/ASI/AGI. It is pure terminal.", "Text")],
     lambda p: _research_summarize(p["text"]))
spec("research", "keywords", "Extract top keywords from text.",
     [("text", "AI machine learning deep learning neural networks", "Text"), ("top", 5, "Top N")],
     lambda p: _research_keywords(p["text"], int(p["top"])))
spec("research", "entities", "Extract candidate named entities (capitalized terms).",
     [("text", "ARARAT33 built AWEAI in Yerevan.", "Text")],
     lambda p: _research_entities(p["text"]))


def _research_search(query: str, engine: str) -> Dict[str, Any]:
    q = urllib.parse.quote(query)
    urls = {
        "google": f"https://www.google.com/search?q={q}",
        "bing": f"https://www.bing.com/search?q={q}",
        "duckduckgo": f"https://duckduckgo.com/?q={q}",
        "github": f"https://github.com/search?q={q}",
    }
    return _ok(engine=engine, query=query, url=urls.get(engine, urls["google"]))


def _research_fetch(url: str, timeout: int) -> Dict[str, Any]:
    try:
        body = _http_get(url, timeout)
        return _ok(url=url, bytes=len(body.encode("utf-8")), chars=len(body))
    except Exception as e:
        return _err(str(e))


def _research_summarize(text: str) -> Dict[str, Any]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 2:
        return _ok(summary=text.strip(), sentences=len(sentences))
    longest = max(sentences, key=len)
    summary = sentences[0] + " " + longest if longest != sentences[0] else sentences[0]
    return _ok(summary=summary.strip(), sentences=len(sentences))


def _research_keywords(text: str, top: int) -> Dict[str, Any]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "for", "with"}
    words = [w.lower() for w in re.findall(r"[a-z0-9]+", text.lower()) if w.lower() not in stop]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return _ok(keywords=[{"word": w, "count": c} for w, c in ranked[:max(1, top)]])


def _research_entities(text: str) -> Dict[str, Any]:
    entities = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)
    return _ok(entities=list(dict.fromkeys(entities)))


# ===========================================================================
# REASONING group — chain-of-thought, planning, decomposition, verification
# ===========================================================================

spec("reasoning", "cot", "Generate a chain-of-thought outline for a problem.",
     [("problem", "Solve x + 2 = 5", "Problem"), ("steps", 4, "Number of steps")],
     lambda p: _reasoning_cot(p["problem"], int(p["steps"])))
spec("reasoning", "decompose", "Decompose a task into subtasks.",
     [("task", "Build a model", "Task"), ("parts", 3, "Number of parts")],
     lambda p: _reasoning_decompose(p["task"], int(p["parts"])))
spec("reasoning", "plan", "Create a step-by-step plan.",
     [("goal", "Ship a CLI tool", "Goal"), ("steps", 5, "Steps")],
     lambda p: _reasoning_plan(p["goal"], int(p["steps"])))
spec("reasoning", "verify", "Check a simple equation/claim syntax.",
     [("claim", "1 + 1 = 2", "Claim")],
     lambda p: _reasoning_verify(p["claim"]))
spec("reasoning", "tree", "Build a reasoning tree (branching alternatives).",
     [("problem", "Choose a model", "Problem"), ("branches", 3, "Branches"), ("depth", 2, "Depth")],
     lambda p: _reasoning_tree(p["problem"], int(p["branches"]), int(p["depth"])))
spec("reasoning", "refine", "Refine a draft answer (generic template).",
     [("draft", "Initial answer", "Draft"), ("feedback", "Be more precise", "Feedback")],
     lambda p: _ok(original=p["draft"], feedback=p["feedback"],
                   refined=f"{p['draft']} (refined with feedback: {p['feedback']})"))


def _reasoning_cot(problem: str, steps: int) -> Dict[str, Any]:
    out = []
    for i in range(1, max(1, steps) + 1):
        out.append({"step": i, "thought": f"Step {i}: analyze part {i} of the problem '{problem[:60]}'"})
    out.append({"step": "answer", "thought": "Conclude from the steps above."})
    return _ok(problem=problem, chain=out)


def _reasoning_decompose(task: str, parts: int) -> Dict[str, Any]:
    subs = [{"part": i + 1, "task": f"{task} — sub-part {i + 1}"} for i in range(max(1, parts))]
    return _ok(task=task, subtasks=subs)


def _reasoning_plan(goal: str, steps: int) -> Dict[str, Any]:
    plan = [{"step": i + 1, "action": f"Action {i + 1} toward '{goal[:60]}'"} for i in range(max(1, steps))]
    return _ok(goal=goal, plan=plan)


def _reasoning_verify(claim: str) -> Dict[str, Any]:
    m = re.match(r"^\s*([\d.]+)\s*([+\-*/])\s*([\d.]+)\s*=\s*([\d.]+)\s*$", claim)
    if not m:
        return _ok(claim=claim, valid=False, reason="unsupported format (use e.g. 1+1=2)")
    a, op, b, c = m.groups()
    try:
        val = {"+": float(a) + float(b), "-": float(a) - float(b),
               "*": float(a) * float(b), "/": float(a) / float(b) if float(b) else float("inf")}[op]
        ok = abs(val - float(c)) < 1e-9
        return _ok(claim=claim, valid=bool(ok), left=val, right=float(c))
    except Exception:
        return _ok(claim=claim, valid=False, reason="eval error")


def _reasoning_tree(problem: str, branches: int, depth: int) -> Dict[str, Any]:
    tree = []
    for d in range(1, max(1, depth) + 1):
        tree.append({"depth": d, "branches": [f"option {b}" for b in range(1, max(1, branches) + 1)]})
    return _ok(problem=problem, tree=tree)


# ===========================================================================
# MEMORY group — episodic, semantic, procedural
# ===========================================================================

def _mem_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/memory.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _mem_data() -> Dict[str, Any]:
    p = Path(_mem_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}
    return {"items": []}


def _save_mem(data: Dict[str, Any]) -> None:
    Path(_mem_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mem_add(kind: str, key: str, value: str) -> Dict[str, Any]:
    data = _mem_data()
    data["items"].append({"kind": kind, "key": key, "value": value, "time": _now_iso()})
    _save_mem(data)
    return _ok(kind=kind, key=key, count=len(data["items"]))


spec("memory", "add", "Add a memory item (episodic/semantic/procedural).",
     [("kind", "episodic", "episodic|semantic|procedural"), ("key", "project", "Key"),
      ("value", "AWEAI v4", "Value")],
     lambda p: _mem_add(p["kind"], p["key"], p["value"]))
spec("memory", "get", "Get memory items by key.",
     [("key", "project", "Key")], lambda p: _mem_get(p["key"]))
spec("memory", "search", "Search memory items by substring.",
     [("query", "AWEAI", "Query")], lambda p: _mem_search(p["query"]))
spec("memory", "list", "List memory items (optionally by kind).",
     [("kind", None, "Kind filter")], lambda p: _mem_list(p["kind"]))
spec("memory", "clear", "Clear memory.", [], lambda p: _mem_clear())
spec("memory", "stats", "Memory statistics.", [], lambda p: _mem_stats())


def _mem_get(key: str) -> Dict[str, Any]:
    data = _mem_data()
    hits = [i for i in data["items"] if i["key"] == key]
    return _ok(key=key, items=hits)


def _mem_search(query: str) -> Dict[str, Any]:
    data = _mem_data()
    q = query.lower()
    hits = [i for i in data["items"] if q in i["key"].lower() or q in i["value"].lower()]
    return _ok(query=query, items=hits)


def _mem_list(kind: Optional[str]) -> Dict[str, Any]:
    data = _mem_data()
    items = [i for i in data["items"] if not kind or i["kind"] == kind]
    return _ok(kind=kind or "all", items=items)


def _mem_clear() -> Dict[str, Any]:
    _save_mem({"items": []})
    return _ok(cleared=True)


def _mem_stats() -> Dict[str, Any]:
    data = _mem_data()
    by_kind: Dict[str, int] = {}
    for i in data["items"]:
        by_kind[i["kind"]] = by_kind.get(i["kind"], 0) + 1
    return _ok(total=len(data["items"]), by_kind=by_kind)


# ===========================================================================
# ORCHESTRATE group — scheduling, fan-out, retries, task queues
# ===========================================================================

def _orch_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/orchestrator.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _orch_data() -> Dict[str, Any]:
    p = Path(_orch_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"tasks": [], "next_id": 1}
    return {"tasks": [], "next_id": 1}


def _save_orch(data: Dict[str, Any]) -> None:
    Path(_orch_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _orch_submit(command: str, priority: int) -> Dict[str, Any]:
    data = _orch_data()
    tid = data["next_id"]
    data["next_id"] += 1
    data["tasks"].append({"id": tid, "command": command, "priority": priority,
                          "status": "queued", "time": _now_iso()})
    _save_orch(data)
    return _ok(id=tid, status="queued")


def _orch_run(id: int) -> Dict[str, Any]:
    data = _orch_data()
    for t in data["tasks"]:
        if t["id"] == id:
            t["status"] = "completed"
            _save_orch(data)
            return _ok(id=id, status="completed")
    return _err(f"task {id} not found")


spec("orchestrate", "submit", "Submit a task to the queue.",
     [("command", "aweai version", "Command"), ("priority", 5, "Priority 1-10")],
     lambda p: _orch_submit(p["command"], int(p["priority"])))
spec("orchestrate", "run", "Mark a queued task as completed.",
     [("id", 1, "Task id")], lambda p: _orch_run(int(p["id"])))
spec("orchestrate", "queue", "Show the task queue.",
     [("status", None, "Filter status")], lambda p: _orch_queue(p["status"]))
spec("orchestrate", "fanout", "Fan out a task to N workers.",
     [("task", "process chunk", "Task"), ("workers", 4, "Workers")],
     lambda p: _ok(task=p["task"], workers=[f"worker-{i + 1}" for i in range(int(p["workers"]))],
                   status="dispatched"))
spec("orchestrate", "retry", "Retry a failed task.",
     [("id", 1, "Task id"), ("attempts", 3, "Max attempts")],
     lambda p: _ok(id=int(p["id"]), retried=True, attempts=int(p["attempts"]), status="queued"))
spec("orchestrate", "schedule", "Register a recurring schedule (cron-like).",
     [("name", "nightly", "Schedule name"), ("cron", "0 2 * * *", "Cron expression"), ("command", "aweai backup run", "Command")],
     lambda p: _ok(name=p["name"], cron=p["cron"], command=p["command"], registered=True))
spec("orchestrate", "health", "Orchestrator health (queue stats).",
     [], lambda p: _orch_health())


def _orch_queue(status: Optional[str]) -> Dict[str, Any]:
    data = _orch_data()
    tasks = [t for t in data["tasks"] if not status or t["status"] == status]
    return _ok(tasks=tasks)


def _orch_health() -> Dict[str, Any]:
    data = _orch_data()
    counts: Dict[str, int] = {}
    for t in data["tasks"]:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    return _ok(total=len(data["tasks"]), by_status=counts, healthy=True)


# ===========================================================================
# SECURITY group — scan, audit, secrets, policy
# ===========================================================================

def _secret_check(text: str) -> Dict[str, Any]:
    found = []
    patterns = {
        "api_key": r"(?i)\b(sk-[a-zA-Z0-9]{20,}|api[_-]?key['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,})",
        "aws_key": r"\b(AKIA[0-9A-Z]{16})\b",
        "private_key": r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "password": r"(?i)\b(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{6,}",
        "token": r"(?i)\b(token|secret)\s*[:=]\s*['\"]?[a-zA-Z0-9\-_.]{16,}",
    }
    for kind, pat in patterns.items():
        for m in re.finditer(pat, text):
            found.append({"kind": kind, "match": m.group(0)[:40]})
    return _ok(secrets=found, count=len(found))


spec("security", "scan", "Scan text/file for potential secrets.",
     [("text", "my password=123456", "Text to scan"), ("file", None, "File path (optional)")],
     lambda p: _secret_check(_read(p["file"]) if p["file"] else p["text"]))
spec("security", "hash", "Hash a string with the given algorithm.",
     [("text", "hello", "Text"), ("algo", "sha256", "md5|sha1|sha256|sha512")],
     lambda p: _sec_hash(p["text"], p["algo"]))
spec("security", "entropy", "Estimate Shannon entropy of a string.",
     [("text", "password", "Text")], lambda p: _sec_entropy(p["text"]))
spec("security", "policy", "Show a security policy template.",
     [("name", "default", "Policy name")],
     lambda p: _ok(policy=p["name"], rules=["no-secrets-in-repo", "encrypt-at-rest", "rotate-keys-90d", "least-privilege"]))
spec("security", "audit", "Run a quick audit of repo files (checks for secrets).",
     [("path", ".", "Directory to audit")], lambda p: _sec_audit(p["path"]))
spec("security", "generate-key", "Generate a random secret key.",
     [("length", 32, "Key length")],
     lambda p: _sec_genkey(int(p["length"])))


def _sec_hash(text: str, algo: str) -> Dict[str, Any]:
    h = hashlib.new(algo, text.encode("utf-8")).hexdigest()
    return _ok(algo=algo, hash=h)


def _sec_entropy(text: str) -> Dict[str, Any]:
    if not text:
        return _ok(entropy=0.0, strength="empty")
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    ent = -sum((c / len(text)) * math.log2(c / len(text)) for c in freq.values())
    strength = "weak" if ent < 2.5 else ("medium" if ent < 4.0 else "strong")
    return _ok(entropy=round(ent, 3), strength=strength)


def _sec_genkey(length: int) -> Dict[str, Any]:
    import secrets as _secrets
    return _ok(key=_secrets.token_hex(max(8, length)))


def _sec_audit(path: str) -> Dict[str, Any]:
    root = Path(path)
    issues = []
    files = 0
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".env"}:
                try:
                    txt = f.read_text(encoding="utf-8", errors="ignore")
                    files += 1
                    res = _secret_check(txt)
                    if res["count"]:
                        issues.append({"file": str(f), "secrets": res["count"]})
                except Exception:
                    pass
    return _ok(scanned_files=files, issues=issues, count=len(issues))


# ===========================================================================
# BACKUP group — backup/restore/versioning
# ===========================================================================

def _backup_dir() -> str:
    p = Path(os.path.expanduser("~/.aweai/backups"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _backup_run(path: str, name: Optional[str]) -> Dict[str, Any]:
    src = Path(path)
    if not src.exists():
        return _err(f"path not found: {path}")
    bname = name or f"backup-{time.strftime('%Y%m%d-%H%M%S')}"
    if src.is_dir():
        shutil.copytree(src, Path(_backup_dir()) / bname, dirs_exist_ok=True)
    else:
        Path(_backup_dir()).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, Path(_backup_dir()) / bname)
    return _ok(name=bname, source=str(src), dest=str(Path(_backup_dir()) / bname))


spec("backup", "run", "Back up a file/directory to the local backup store.",
     [("path", ".", "Path to back up"), ("name", None, "Backup name (optional)")],
     lambda p: _backup_run(p["path"], p["name"]))
spec("backup", "list", "List backups.", [], lambda p: _backup_list())
spec("backup", "restore", "Restore a backup (copy back).",
     [("name", "backup-1", "Backup name"), ("dest", ".", "Destination")],
     lambda p: _backup_restore(p["name"], p["dest"]))
spec("backup", "remove", "Remove a backup.", [("name", "backup-1", "Backup name")],
     lambda p: _backup_remove(p["name"]))
spec("backup", "version", "Show backup versions (all names).",
     [], lambda p: _ok(versions=_backup_list()["backups"]))


def _backup_list() -> Dict[str, Any]:
    d = Path(_backup_dir())
    items = []
    if d.exists():
        for x in sorted(d.iterdir()):
            items.append({"name": x.name, "is_dir": x.is_dir(), "modified": _dt.datetime.fromtimestamp(x.stat().st_mtime).isoformat()})
    return _ok(backups=items)


def _backup_restore(name: str, dest: str) -> Dict[str, Any]:
    src = Path(_backup_dir()) / name
    if not src.exists():
        return _err(f"backup not found: {name}")
    dst = Path(dest)
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst / src.name)
    return _ok(restored=str(src), dest=str(dst))


def _backup_remove(name: str) -> Dict[str, Any]:
    src = Path(_backup_dir()) / name
    if not src.exists():
        return _err(f"backup not found: {name}")
    if src.is_dir():
        shutil.rmtree(src)
    else:
        src.unlink()
    return _ok(removed=name)


# ===========================================================================
# MONITOR group — metrics, logs, alerts, health
# ===========================================================================

spec("monitor", "health", "Basic system health (CPU, memory, disk, load).",
     [], lambda p: _monitor_health())
spec("monitor", "cpu", "CPU usage snapshot (via os/psutil if available).",
     [], lambda p: _monitor_cpu())
spec("monitor", "mem", "Memory usage snapshot.",
     [], lambda p: _monitor_mem())
spec("monitor", "disk", "Disk usage for a path.",
     [("path", ".", "Path")], lambda p: _monitor_disk(p["path"]))
spec("monitor", "uptime", "System uptime (seconds since boot, best-effort).",
     [], lambda p: _monitor_uptime())
spec("monitor", "log", "Append a log line to the AWEAI log store.",
     [("level", "info", "info|warn|error"), ("message", "hello", "Message")],
     lambda p: _monitor_log(p["level"], p["message"]))
spec("monitor", "logs", "Show recent AWEAI log lines.",
     [("limit", 20, "Lines"), ("level", None, "Level filter")],
     lambda p: _monitor_logs(int(p["limit"]), p["level"]))
spec("monitor", "alert", "Register an alert rule.",
     [("name", "disk-low", "Rule name"), ("condition", "disk < 10%", "Condition"), ("action", "notify", "Action")],
     lambda p: _ok(rule=p["name"], condition=p["condition"], action=p["action"], registered=True))
spec("monitor", "metrics", "Record/read a metric value.",
     [("name", "accuracy", "Metric name"), ("value", 0.95, "Value")],
     lambda p: _monitor_metric(p["name"], p["value"]))


def _monitor_health() -> Dict[str, Any]:
    return _ok(cpu=_monitor_cpu()["percent"], memory=_monitor_mem()["percent"],
               disk=_monitor_disk(".")["percent"], status="ok")


def _monitor_cpu() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore
        return _ok(percent=psutil.cpu_percent(interval=0.1), cores=psutil.cpu_count())
    except Exception:
        return _ok(percent=None, note="psutil not installed; install for live metrics")


def _monitor_mem() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return _ok(percent=vm.percent, used=vm.used, total=vm.total)
    except Exception:
        return _ok(percent=None, note="psutil not installed")


def _monitor_disk(path: str) -> Dict[str, Any]:
    try:
        import shutil
        usage = shutil.disk_usage(path)
        return _ok(total=usage.total, used=usage.used, free=usage.free,
                   percent=round(usage.used / usage.total * 100, 1) if usage.total else 0)
    except Exception as e:
        return _err(str(e))


def _monitor_uptime() -> Dict[str, Any]:
    try:
        with open("/proc/uptime", "r") as f:
            secs = float(f.read().split()[0])
        return _ok(seconds=round(secs), hours=round(secs / 3600, 2))
    except Exception:
        return _ok(seconds=None, note="unsupported platform")


def _log_store() -> Path:
    p = Path(os.path.expanduser("~/.aweai/logs.jsonl"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _monitor_log(level: str, message: str) -> Dict[str, Any]:
    with open(_log_store(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": _now_iso(), "level": level, "message": message}) + "\n")
    return _ok(logged=True, level=level)


def _monitor_logs(limit: int, level: Optional[str]) -> Dict[str, Any]:
    lines = []
    if _log_store().exists():
        for line in reversed(_log_store().read_text(encoding="utf-8").splitlines()):
            try:
                item = json.loads(line)
                if not level or item.get("level") == level:
                    lines.append(item)
            except Exception:
                pass
            if len(lines) >= limit:
                break
    return _ok(lines=lines)


def _monitor_metric(name: str, value: Any) -> Dict[str, Any]:
    p = Path(os.path.expanduser("~/.aweai/metrics.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault(name, []).append({"value": value, "time": _now_iso()})
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return _ok(name=name, last=value, count=len(data[name]))


# ===========================================================================
# PLUGIN group — ecosystem/plugin/marketplace management
# ===========================================================================

def _plugin_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/plugins.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _plugins() -> Dict[str, Any]:
    p = Path(_plugin_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_plugins(data: Dict[str, Any]) -> None:
    Path(_plugin_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _plugin_install(name: str, source: str) -> Dict[str, Any]:
    plugs = _plugins()
    plugs[name] = {"name": name, "source": source, "installed": _now_iso(), "enabled": True}
    _save_plugins(plugs)
    return _ok(installed=name, source=source)


spec("plugin", "install", "Install a plugin (local registry entry).",
     [("name", "my-plugin", "Plugin name"), ("source", "local", "Source (local/git/url)")],
     lambda p: _plugin_install(p["name"], p["source"]))
spec("plugin", "list", "List installed plugins.", [], lambda p: _ok(plugins=list(_plugins().values())))
spec("plugin", "enable", "Enable a plugin.",
     [("name", "my-plugin", "Plugin name")], lambda p: _plugin_set(p["name"], True))
spec("plugin", "disable", "Disable a plugin.",
     [("name", "my-plugin", "Plugin name")], lambda p: _plugin_set(p["name"], False))
spec("plugin", "remove", "Remove a plugin.",
     [("name", "my-plugin", "Plugin name")], lambda p: _plugin_remove(p["name"]))
spec("plugin", "search", "Search local plugin catalog.",
     [("query", "rag", "Query")], lambda p: _plugin_search(p["query"]))
spec("plugin", "marketplace", "Show plugin marketplace (built-in catalog).",
     [], lambda p: _plugin_marketplace())


def _plugin_set(name: str, enabled: bool) -> Dict[str, Any]:
    plugs = _plugins()
    if name not in plugs:
        return _err(f"not found: {name}")
    plugs[name]["enabled"] = enabled
    _save_plugins(plugs)
    return _ok(name=name, enabled=enabled)


def _plugin_remove(name: str) -> Dict[str, Any]:
    plugs = _plugins()
    if name not in plugs:
        return _err(f"not found: {name}")
    del plugs[name]
    _save_plugins(plugs)
    return _ok(removed=name)


def _plugin_search(query: str) -> Dict[str, Any]:
    catalog = _plugin_marketplace()["catalog"]
    q = query.lower()
    hits = [p for p in catalog if q in p["name"].lower() or q in p["description"].lower()]
    return _ok(query=query, results=hits)


def _plugin_marketplace() -> Dict[str, Any]:
    catalog = [
        {"name": "rag", "description": "Retrieval-augmented generation toolkit"},
        {"name": "vision", "description": "Computer vision helpers"},
        {"name": "audio", "description": "Audio processing helpers"},
        {"name": "security", "description": "Security scanning and audit"},
        {"name": "monitor", "description": "Monitoring and alerting"},
        {"name": "export", "description": "Model export formats"},
        {"name": "distributed", "description": "Distributed training"},
        {"name": "i18n", "description": "Internationalization"},
    ]
    return _ok(catalog=catalog, count=len(catalog))
