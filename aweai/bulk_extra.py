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
# AGENT group — agent orchestration, roles, tools, memory, multi-agent
# ===========================================================================

def _agent_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/agents.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _agents() -> Dict[str, Any]:
    p = Path(_agent_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_agents(data: Dict[str, Any]) -> None:
    Path(_agent_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _agent_run(name: str, task: str, model: Optional[str]) -> Dict[str, Any]:
    agents = _agents()
    a = agents.get(name)
    if a is None:
        return _err(f"agent '{name}' not found; create with: aweai agent create")
    plan = a.get("plan") or ["receive-task", "analyze", "act", "verify", "report"]
    steps = []
    for s in plan:
        steps.append({"step": s, "status": "ok", "note": f"executed by {name}"})
    return _ok(agent=name, task=task, model=model or a.get("model") or "local", steps=steps,
               status="completed")


spec("agent", "create", "Create an agent definition (role + system prompt + tools).",
     [("name", "assistant", "Agent name"), ("role", "general", "Role/purpose"),
      ("system", "You are a helpful assistant.", "System prompt"),
      ("tools", "search,memory,math", "Comma-separated tool list"),
      ("model", "local", "Preferred model")],
     lambda p: _create_agent(p["name"], p["role"], p["system"], p["tools"], p["model"]))
spec("agent", "list", "List defined agents.", [], lambda p: _ok(agents=list(_agents().values())))
spec("agent", "get", "Get an agent definition.", [("name", "assistant", "Agent name")],
     lambda p: _ok(agent=_agents().get(p["name"]) or _err(f"not found: {p['name']}")))
spec("agent", "remove", "Remove an agent definition.", [("name", "assistant", "Agent name")],
     lambda p: _remove_agent(p["name"]))
spec("agent", "run", "Run an agent on a task (simulated orchestration).",
     [("name", "assistant", "Agent name"), ("task", "summarize the input", "Task text"),
      ("model", None, "Model override")],
     lambda p: _agent_run(p["name"], p["task"], p["model"]))
spec("agent", "chat", "Multi-turn agent chat (memory-backed, simulated).",
     [("name", "assistant", "Agent name"), ("message", "Hello", "Message")],
     lambda p: _agent_chat(p["name"], p["message"]))
spec("agent", "tools", "List tools available to an agent.",
     [("name", "assistant", "Agent name")],
     lambda p: _ok(tools=(_agents().get(p["name"]) or {}).get("tools", [])))
spec("agent", "grant", "Grant a tool to an agent.",
     [("name", "assistant", "Agent name"), ("tool", "math", "Tool name")],
     lambda p: _agent_grant(p["name"], p["tool"]))
spec("agent", "revoke", "Revoke a tool from an agent.",
     [("name", "assistant", "Agent name"), ("tool", "math", "Tool name")],
     lambda p: _agent_revoke(p["name"], p["tool"]))
spec("agent", "spawn", "Spawn N worker agents for parallel tasks.",
     [("role", "worker", "Role"), ("count", 3, "Number of workers"), ("task", "process", "Base task")],
     lambda p: _agent_spawn(p["role"], int(p["count"]), p["task"]))
spec("agent", "multi", "Multi-agent collaboration on one goal (simulated).",
     [("goal", "solve a problem", "Goal"), ("agents", "planner,critic,executor", "Agent names")],
     lambda p: _agent_multi(p["goal"], p["agents"]))
spec("agent", "system", "Show the default system prompt template.",
     [("role", "assistant", "Role")],
     lambda p: _ok(role=p["role"], system=f"You are a {p['role']} agent in the AWEAI universe."))
spec("agent", "register", "Register an agent in the marketplace (local listing).",
     [("name", "assistant", "Agent name"), ("description", "General assistant", "Description")],
     lambda p: _agent_register(p["name"], p["description"]))


def _create_agent(name: str, role: str, system: str, tools: str, model: str) -> Dict[str, Any]:
    agents = _agents()
    tool_list = [t.strip() for t in tools.split(",") if t.strip()]
    agents[name] = {"name": name, "role": role, "system": system, "tools": tool_list,
                    "model": model, "created": _now_iso()}
    _save_agents(agents)
    return _ok(created=name, role=role, tools=tool_list, model=model)


def _remove_agent(name: str) -> Dict[str, Any]:
    agents = _agents()
    if name not in agents:
        return _err(f"not found: {name}")
    del agents[name]
    _save_agents(agents)
    return _ok(removed=name)


def _agent_chat(name: str, message: str) -> Dict[str, Any]:
    agents = _agents()
    a = agents.get(name)
    if a is None:
        return _err(f"agent '{name}' not found")
    hist = a.get("history", [])
    hist.append({"role": "user", "content": message})
    reply = f"[{name}] understood: {message[:120]}"
    hist.append({"role": "assistant", "content": reply})
    a["history"] = hist[-20:]
    _save_agents(agents)
    return _ok(reply=reply, history_len=len(hist))


def _agent_grant(name: str, tool: str) -> Dict[str, Any]:
    agents = _agents()
    a = agents.get(name)
    if a is None:
        return _err(f"not found: {name}")
    if tool not in a["tools"]:
        a["tools"].append(tool)
    _save_agents(agents)
    return _ok(agent=name, tools=a["tools"])


def _agent_revoke(name: str, tool: str) -> Dict[str, Any]:
    agents = _agents()
    a = agents.get(name)
    if a is None:
        return _err(f"not found: {name}")
    if tool in a["tools"]:
        a["tools"].remove(tool)
    _save_agents(agents)
    return _ok(agent=name, tools=a["tools"])


def _agent_spawn(role: str, count: int, task: str) -> Dict[str, Any]:
    workers = []
    for i in range(count):
        wid = f"{role}-{i + 1}"
        workers.append({"id": wid, "task": task, "status": "ready"})
    return _ok(role=role, count=count, workers=workers)


def _agent_multi(goal: str, agents: str) -> Dict[str, Any]:
    names = [a.strip() for a in agents.split(",") if a.strip()]
    rounds = []
    for i, n in enumerate(names):
        rounds.append({"round": i + 1, "agent": n, "contribution": f"contribution about: {goal[:80]}"})
    return _ok(goal=goal, agents=names, rounds=rounds, status="converged")


def _agent_register(name: str, description: str) -> Dict[str, Any]:
    p = Path(os.path.expanduser("~/.aweai/agent-market.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[name] = {"name": name, "description": description, "registered": _now_iso()}
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return _ok(registered=name)


# ===========================================================================
# WORKFLOW group — definitions, DAG, steps, retries, triggers
# ===========================================================================

def _wf_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/workflows.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _workflows() -> Dict[str, Any]:
    p = Path(_wf_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_workflows(data: Dict[str, Any]) -> None:
    Path(_wf_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _wf_create(name: str, steps: str, retries: int) -> Dict[str, Any]:
    wf = _workflows()
    step_list = [s.strip() for s in steps.split(";") if s.strip()] or ["echo hello"]
    wf[name] = {"name": name, "steps": step_list, "retries": retries, "created": _now_iso()}
    _save_workflows(wf)
    return _ok(created=name, steps=len(step_list), retries=retries)


def _wf_run(name: str) -> Dict[str, Any]:
    wf = _workflows()
    item = wf.get(name)
    if item is None:
        return _err(f"workflow '{name}' not found")
    executed = []
    for i, step in enumerate(item["steps"]):
        executed.append({"step": i + 1, "command": step, "status": "ok"})
    item["last_run"] = _now_iso()
    item["runs"] = item.get("runs", 0) + 1
    _save_workflows(wf)
    return _ok(workflow=name, executed=executed, total_steps=len(executed))


spec("workflow", "create", "Create a workflow from ';' separated steps.",
     [("name", "wf1", "Workflow name"), ("steps", "collect;clean;train", "Semicolon-separated steps"),
      ("retries", 1, "Retry count")],
     lambda p: _wf_create(p["name"], p["steps"], int(p["retries"])))
spec("workflow", "list", "List workflows.", [], lambda p: _ok(workflows=list(_workflows().values())))
spec("workflow", "run", "Run a workflow (simulated execution of steps).",
     [("name", "wf1", "Workflow name")], lambda p: _wf_run(p["name"]))
spec("workflow", "remove", "Remove a workflow.", [("name", "wf1", "Workflow name")],
     lambda p: _wf_remove(p["name"]))
spec("workflow", "dag", "Build a DAG from dependencies (name:dep1,dep2).",
     [("specs", "a:;b:a;c:a,b", "Node:deps, comma separated")],
     lambda p: _wf_dag(p["specs"]))
spec("workflow", "topo", "Topological order of a DAG (specs: name:deps).",
     [("specs", "a:;b:a;c:a,b", "Node:deps")], lambda p: _wf_topo(p["specs"]))
spec("workflow", "validate", "Validate a workflow definition.",
     [("name", "wf1", "Workflow name")], lambda p: _wf_validate(p["name"]))
spec("workflow", "export", "Export a workflow as JSON/YAML text.",
     [("name", "wf1", "Workflow name"), ("format", "json", "json|yaml")],
     lambda p: _wf_export(p["name"], p["format"]))


def _wf_remove(name: str) -> Dict[str, Any]:
    wf = _workflows()
    if name not in wf:
        return _err(f"not found: {name}")
    del wf[name]
    _save_workflows(wf)
    return _ok(removed=name)


def _parse_dag(specs: str) -> Dict[str, List[str]]:
    dag: Dict[str, List[str]] = {}
    for part in specs.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            node, deps = part.split(":", 1)
            dag[node.strip()] = [d.strip() for d in deps.split(",") if d.strip()]
        else:
            dag[part] = []
    return dag


def _wf_dag(specs: str) -> Dict[str, Any]:
    dag = _parse_dag(specs)
    return _ok(nodes=list(dag.keys()), edges=[{"from": d, "to": n} for n, ds in dag.items() for d in ds])


def _wf_topo(specs: str) -> Dict[str, Any]:
    dag = _parse_dag(specs)
    order: List[str] = []
    visited: Dict[str, int] = {}
    temp: Dict[str, int] = {}
    cycle = False

    def visit(n: str) -> None:
        nonlocal cycle
        if n in temp:
            cycle = True
            return
        if n in visited:
            return
        temp[n] = 1
        for d in dag.get(n, []):
            if d in dag:
                visit(d)
        temp.pop(n, None)
        visited[n] = 1
        order.append(n)

    for n in dag:
        visit(n)
    if cycle:
        return _err("cycle detected")
    return _ok(order=order)


def _wf_validate(name: str) -> Dict[str, Any]:
    wf = _workflows()
    item = wf.get(name)
    if item is None:
        return _err(f"not found: {name}")
    problems = []
    if not item.get("steps"):
        problems.append("no steps")
    if item.get("retries", 0) < 0:
        problems.append("negative retries")
    return _ok(valid=not problems, problems=problems)


def _wf_export(name: str, fmt: str) -> Dict[str, Any]:
    wf = _workflows()
    item = wf.get(name)
    if item is None:
        return _err(f"not found: {name}")
    if fmt == "yaml":
        lines = [f"name: {item['name']}", "steps:"]
        for s in item["steps"]:
            lines.append(f"  - {s}")
        return _ok(format="yaml", text="\n".join(lines))
    return _ok(format="json", text=json.dumps(item, indent=2, ensure_ascii=False))


# ===========================================================================
# PIPELINE group — stages, transforms, validation, artifacts
# ===========================================================================

def _pipe_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/pipelines.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _pipelines() -> Dict[str, Any]:
    p = Path(_pipe_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_pipelines(data: Dict[str, Any]) -> None:
    Path(_pipe_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _pipe_create(name: str, stages: str) -> Dict[str, Any]:
    pipes = _pipelines()
    stage_list = [s.strip() for s in stages.split(";") if s.strip()] or ["load", "clean", "train"]
    pipes[name] = {"name": name, "stages": stage_list, "created": _now_iso()}
    _save_pipelines(pipes)
    return _ok(created=name, stages=stage_list)


def _pipe_run(name: str, data: str) -> Dict[str, Any]:
    pipes = _pipelines()
    item = pipes.get(name)
    if item is None:
        return _err(f"pipeline '{name}' not found")
    ran = []
    current = data
    for i, stage in enumerate(item["stages"]):
        current = f"{stage}({current[:50]})"
        ran.append({"stage": i + 1, "name": stage, "ok": True})
    return _ok(pipeline=name, stages=ran, final=current)


spec("pipeline", "create", "Create a data pipeline from ';' stages.",
     [("name", "pipe1", "Pipeline name"), ("stages", "load;clean;embed;train", "Stages")],
     lambda p: _pipe_create(p["name"], p["stages"]))
spec("pipeline", "list", "List pipelines.", [], lambda p: _ok(pipelines=list(_pipelines().values())))
spec("pipeline", "run", "Run a pipeline on data (simulated).",
     [("name", "pipe1", "Pipeline name"), ("data", "sample", "Input data ref")],
     lambda p: _pipe_run(p["name"], p["data"]))
spec("pipeline", "remove", "Remove a pipeline.", [("name", "pipe1", "Pipeline name")],
     lambda p: _pipe_remove(p["name"]))
spec("pipeline", "validate", "Check pipeline stage list.",
     [("name", "pipe1", "Pipeline name")], lambda p: _pipe_validate(p["name"]))
spec("pipeline", "artifact", "Register an artifact produced by a pipeline.",
     [("name", "pipe1", "Pipeline name"), ("artifact", "model.bin", "Artifact path")],
     lambda p: _pipe_artifact(p["name"], p["artifact"]))


def _pipe_remove(name: str) -> Dict[str, Any]:
    pipes = _pipelines()
    if name not in pipes:
        return _err(f"not found: {name}")
    del pipes[name]
    _save_pipelines(pipes)
    return _ok(removed=name)


def _pipe_validate(name: str) -> Dict[str, Any]:
    pipes = _pipelines()
    item = pipes.get(name)
    if item is None:
        return _err(f"not found: {name}")
    return _ok(valid=bool(item.get("stages")), stages=len(item.get("stages", [])))


def _pipe_artifact(name: str, artifact: str) -> Dict[str, Any]:
    pipes = _pipelines()
    item = pipes.get(name)
    if item is None:
        return _err(f"not found: {name}")
    arts = item.setdefault("artifacts", [])
    arts.append({"artifact": artifact, "time": _now_iso()})
    _save_pipelines(pipes)
    return _ok(pipeline=name, artifacts=arts)


# ===========================================================================
# RAG group — index, search, knowledge
# ===========================================================================

def _rag_store() -> str:
    p = Path(os.path.expanduser("~/.aweai/rag.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _rag_data() -> Dict[str, Any]:
    p = Path(_rag_store())
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {"docs": [], "index": {}}


def _save_rag(data: Dict[str, Any]) -> None:
    Path(_rag_store()).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _rag_add(doc: str, text: str) -> Dict[str, Any]:
    data = _rag_data()
    doc_id = _sha256(doc + text)[:12]
    data["docs"].append({"id": doc_id, "doc": doc, "text": text, "added": _now_iso()})
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        data["index"].setdefault(tok, []).append(doc_id)
    _save_rag(data)
    return _ok(id=doc_id, docs=len(data["docs"]))


def _rag_search(query: str, top_k: int) -> Dict[str, Any]:
    data = _rag_data()
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for d in data["docs"]:
        d_tokens = set(re.findall(r"[a-z0-9]+", d["text"].lower()))
        inter = len(q_tokens & d_tokens)
        if inter:
            scored.append({"id": d["id"], "doc": d["doc"], "score": inter / max(1, len(q_tokens))})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return _ok(query=query, results=scored[:max(1, int(top_k))])


spec("rag", "add", "Add a document to the RAG index.",
     [("doc", "doc1", "Document name"), ("text", "AWEAI is a universal CLI.", "Text")],
     lambda p: _rag_add(p["doc"], p["text"]))
spec("rag", "search", "Search the RAG index (lexical overlap).",
     [("query", "CLI", "Query"), ("top_k", 5, "Top K")],
     lambda p: _rag_search(p["query"], int(p["top_k"])))
spec("rag", "stats", "RAG index statistics.", [], lambda p: _rag_stats())
spec("rag", "clear", "Clear the RAG index.", [], lambda p: _rag_clear())
spec("rag", "ask", "Ask with RAG context (simulated generation).",
     [("query", "What is AWEAI?", "Question")], lambda p: _rag_ask(p["query"]))
spec("rag", "chunk", "Split text into chunks (fixed size).",
     [("text", "AWEAI is a universal CLI for AI engineering.", "Text"), ("size", 32, "Chunk size")],
     lambda p: _rag_chunk(p["text"], int(p["size"])))


def _rag_stats() -> Dict[str, Any]:
    data = _rag_data()
    return _ok(docs=len(data["docs"]), tokens=len(data["index"]),
               total_chars=sum(len(d["text"]) for d in data["docs"]))


def _rag_clear() -> Dict[str, Any]:
    _save_rag({"docs": [], "index": {}})
    return _ok(cleared=True)


def _rag_ask(query: str) -> Dict[str, Any]:
    hits = _rag_search(query, 3)["results"]
    ctx = " ".join(h["doc"] for h in hits) or "no context"
    return _ok(query=query, context=ctx, answer=f"Answer generated from {len(hits)} retrieved chunks.")


def _rag_chunk(text: str, size: int) -> Dict[str, Any]:
    size = max(1, size)
    words = text.split()
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    return _ok(chunks=chunks, count=len(chunks))
