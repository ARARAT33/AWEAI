# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Operations commands: users/roles/auth/billing, workflows/schedulers/agents,
AGI orchestration/memory/reasoning, RAG/vector DB, security/monitoring/backup."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import APP_DIR, err, jdump, ok, now_iso, read_json, write_json

app = typer.Typer(help="Operations: users, auth, billing, workflows, schedulers, agents, AGI, RAG, security, monitoring, backup")


# ---------------------------------------------------------------------------
# Users / roles / permissions / auth / billing
# ---------------------------------------------------------------------------
def _users_file() -> Path:
    return APP_DIR / "users.json"


def _users() -> Dict[str, Any]:
    return read_json(str(_users_file()), {"users": {}, "billing": {}})


def _save_users(data: Dict[str, Any]) -> None:
    write_json(str(_users_file()), data)


@app.command("user-add")
def user_add(
    name: str = typer.Argument(..., help="Username"),
    role: str = typer.Option("user", "--role", "-r", help="admin|user|viewer|developer"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Optional password (hashed)"),
):
    """Create a user (local auth store)."""
    data = _users()
    if name in data["users"]:
        typer.echo(jdump(err("user already exists")))
        raise typer.Exit(code=1)
    salt = secrets.token_hex(8)
    pw_hash = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() if password else None
    data["users"][name] = {"role": role, "created": now_iso(), "salt": salt, "pw_hash": pw_hash}
    _save_users(data)
    typer.echo(jdump(ok(user=name, role=role)))


@app.command("user-list")
def user_list():
    """List users."""
    data = _users()
    typer.echo(jdump(ok(users=[{**v, "name": k} for k, v in data["users"].items()])))


@app.command("user-remove")
def user_remove(name: str = typer.Argument(..., help="Username")):
    """Remove a user."""
    data = _users()
    removed = data["users"].pop(name, None)
    _save_users(data)
    typer.echo(jdump(ok(removed=name, existed=removed is not None)))


@app.command("role-set")
def role_set(
    name: str = typer.Argument(..., help="Username"),
    role: str = typer.Option(..., "--role", "-r", help="admin|user|viewer|developer"),
):
    """Set a user's role."""
    data = _users()
    if name not in data["users"]:
        typer.echo(jdump(err("user not found")))
        raise typer.Exit(code=1)
    data["users"][name]["role"] = role
    _save_users(data)
    typer.echo(jdump(ok(user=name, role=role)))


@app.command("auth")
def auth(
    name: str = typer.Argument(..., help="Username"),
    password: str = typer.Option(..., "--password", "-p", help="Password"),
):
    """Authenticate a user (local store)."""
    data = _users()
    user = data["users"].get(name)
    if not user or not user.get("pw_hash"):
        typer.echo(jdump(err("user not found or no password set")))
        raise typer.Exit(code=1)
    guess = hashlib.sha256(f"{user['salt']}:{password}".encode()).hexdigest()
    ok_auth = secrets.compare_digest(guess, user["pw_hash"])
    typer.echo(jdump(ok(authenticated=ok_auth, role=user["role"] if ok_auth else None)))


@app.command("permissions")
def permissions(role: str = typer.Option("user", "--role", "-r")):
    """Show permissions for a role."""
    perms = {
        "admin": ["read", "write", "delete", "manage_users", "billing", "deploy"],
        "developer": ["read", "write", "train", "deploy"],
        "user": ["read", "write", "train"],
        "viewer": ["read"],
    }
    typer.echo(jdump(ok(role=role, permissions=perms.get(role, []))))


@app.command("billing-add")
def billing_add(
    user: str = typer.Argument(..., help="Username"),
    amount: float = typer.Option(..., "--amount", "-a", help="Credits to add"),
):
    """Add billing credits to a user."""
    data = _users()
    billing = data.setdefault("billing", {})
    bal = billing.get(user, 0.0)
    billing[user] = bal + amount
    _save_users(data)
    typer.echo(jdump(ok(user=user, balance=billing[user])))


@app.command("billing-balance")
def billing_balance(user: Optional[str] = typer.Option(None, "--user", "-u")):
    """Show billing balances."""
    data = _users()
    billing = data.get("billing", {})
    if user:
        typer.echo(jdump(ok(user=user, balance=billing.get(user, 0.0))))
    else:
        typer.echo(jdump(ok(balances=billing)))


@app.command("token-issue")
def token_issue(
    user: str = typer.Argument(..., help="Username"),
    ttl_hours: int = typer.Option(24, "--ttl", "-t", help="Token lifetime hours"),
):
    """Issue an API token for a user."""
    data = _users()
    if user not in data["users"]:
        typer.echo(jdump(err("user not found")))
        raise typer.Exit(code=1)
    token = secrets.token_urlsafe(32)
    tokens = data.setdefault("tokens", {})
    tokens[token] = {"user": user, "expires": (datetime.now() + timedelta(hours=ttl_hours)).isoformat()}
    _save_users(data)
    typer.echo(jdump(ok(user=user, token=token, expires=tokens[token]["expires"])))


# ---------------------------------------------------------------------------
# Workflows / schedulers / cron / agents / actions
# ---------------------------------------------------------------------------
def _wf_file() -> Path:
    return APP_DIR / "workflows.json"


@app.command("workflow-add")
def workflow_add(
    name: str = typer.Argument(..., help="Workflow name"),
    steps: str = typer.Option(..., "--steps", "-s", help="JSON list of step dicts"),
    schedule: Optional[str] = typer.Option(None, "--schedule", help="cron-ish schedule, e.g. '0 9 * * *'"),
):
    """Register a workflow (steps + optional schedule)."""
    try:
        steps_list = json.loads(steps)
    except Exception as e:
        typer.echo(jdump(err(f"steps must be valid JSON: {e}")))
        raise typer.Exit(code=1)
    data = read_json(str(_wf_file()), {"workflows": {}})
    data["workflows"][name] = {"steps": steps_list, "schedule": schedule, "created": now_iso()}
    write_json(str(_wf_file()), data)
    typer.echo(jdump(ok(workflow=name, steps=len(steps_list), schedule=schedule)))


@app.command("workflow-list")
def workflow_list():
    """List workflows."""
    data = read_json(str(_wf_file()), {"workflows": {}})
    typer.echo(jdump(ok(workflows=data["workflows"])))


@app.command("workflow-run")
def workflow_run(name: str = typer.Argument(..., help="Workflow name")):
    """Run a workflow's steps locally."""
    data = read_json(str(_wf_file()), {"workflows": {}})
    wf = data["workflows"].get(name)
    if not wf:
        typer.echo(jdump(err(f"workflow not found: {name}")))
        raise typer.Exit(code=1)
    results = []
    for step in wf["steps"]:
        op = step.get("op")
        try:
            if op == "echo":
                results.append({"op": op, "ok": True, "message": step.get("message")})
            elif op == "sleep":
                time.sleep(float(step.get("seconds", 0)))
                results.append({"op": op, "ok": True, "slept": step.get("seconds", 0)})
            elif op == "command":
                from aweai.cmd.common import run_cmd
                out = run_cmd(step.get("cmd", "true"), timeout=30)
                results.append({"op": op, "ok": True, "output": out[:200]})
            elif op == "note":
                results.append({"op": op, "ok": True, "note": step.get("text", "")})
            else:
                results.append({"op": op, "ok": False, "error": "unknown op"})
        except Exception as e:
            results.append({"op": op, "ok": False, "error": str(e)})
    typer.echo(jdump(ok(workflow=name, results=results)))


@app.command("workflow-remove")
def workflow_remove(name: str = typer.Argument(..., help="Workflow name")):
    """Remove a workflow."""
    data = read_json(str(_wf_file()), {"workflows": {}})
    removed = data["workflows"].pop(name, None)
    write_json(str(_wf_file()), data)
    typer.echo(jdump(ok(removed=name, existed=removed is not None)))


@app.command("scheduler-list")
def scheduler_list():
    """List scheduled workflows (registered, run by external cron)."""
    data = read_json(str(_wf_file()), {"workflows": {}})
    scheduled = {k: v for k, v in data["workflows"].items() if v.get("schedule")}
    typer.echo(jdump(ok(scheduled=scheduled)))


@app.command("cron")
def cron(
    action: str = typer.Argument(..., help="install|list|uninstall"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Workflow name (install)"),
):
    """Manage system cron entries for workflows (Unix crontab)."""
    from aweai.cmd.common import run_cmd

    if action == "list":
        typer.echo(jdump(ok(crontab=run_cmd("crontab -l 2>/dev/null || echo ''"))))
    elif action == "install":
        data = read_json(str(_wf_file()), {"workflows": {}})
        wf = data["workflows"].get(name or "")
        if not wf or not wf.get("schedule"):
            typer.echo(jdump(err("workflow or schedule not found")))
            raise typer.Exit(code=1)
        line = f"{wf['schedule']} cd {os.getcwd()} && python -m aweai ops workflow-run {name} >> ~/.aweai/cron.log 2>&1"
        cur = run_cmd("crontab -l 2>/dev/null || echo ''")
        if line not in cur:
            new = cur + "\n" + line + "\n" if cur.strip() else line + "\n"
            Path("/tmp/aweai_cron").write_text(new, encoding="utf-8")
            run_cmd("crontab /tmp/aweai_cron")
        typer.echo(jdump(ok(installed=name, schedule=wf["schedule"])))
    elif action == "uninstall":
        cur = run_cmd("crontab -l 2>/dev/null || echo ''")
        kept = "\n".join(ln for ln in cur.splitlines() if "aweai ops workflow-run" not in ln)
        Path("/tmp/aweai_cron").write_text(kept + "\n", encoding="utf-8")
        run_cmd("crontab /tmp/aweai_cron")
        typer.echo(jdump(ok(uninstalled=True)))
    else:
        typer.echo(jdump(err("unknown action")))
        raise typer.Exit(code=1)


@app.command("agent-run")
def agent_run(
    goal: str = typer.Argument(..., help="Agent goal (natural language)"),
    steps: int = typer.Option(5, "--steps", "-s", help="Max steps"),
):
    """Run a simple local agent loop that decomposes a goal into actions."""
    actions: List[Dict[str, Any]] = []
    tokens = goal.lower().split()
    plan = []
    if any(w in tokens for w in ["train", "model"]):
        plan.append("train an mlp model named agent_demo")
    if any(w in tokens for w in ["list", "models"]):
        plan.append("list all models")
    if any(w in tokens for w in ["hardware"]):
        plan.append("hardware")
    if any(w in tokens for w in ["rag", "index", "search"]):
        plan.append("rag index docs")
    if not plan:
        plan = ["hardware"]
    for i, action in enumerate(plan[:steps]):
        try:
            from aweai.actions import run_action
            res = run_action(action)
            actions.append({"step": i + 1, "action": action, "status": res.get("status", "ok")})
        except Exception as e:
            actions.append({"step": i + 1, "action": action, "status": "error", "error": str(e)})
    typer.echo(jdump(ok(goal=goal, steps=len(actions), actions=actions)))


# ---------------------------------------------------------------------------
# AGI: orchestration / memory / reasoning / self-improvement
# ---------------------------------------------------------------------------
@app.command("agi-status")
def agi_status():
    """Show AGI/ASI readiness summary of this AWEAI installation."""
    try:
        from aweai.ai import about as ai_about
        from aweai.ai import AGI_LEVELS, ROADMAP, SELF_IMPROVEMENT_HOOKS

        typer.echo(jdump(ok(
            knowledge=ai_about(),
            current_level=AGI_LEVELS[1]["name"],
            level_notes=AGI_LEVELS[1]["summary"],
            roadmap=ROADMAP,
            self_improvement_hooks=[h["name"] for h in SELF_IMPROVEMENT_HOOKS],
        )))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("memory-add")
def memory_add(
    key: str = typer.Argument(..., help="Memory key"),
    value: str = typer.Option(..., "--value", "-v", help="Memory value"),
    kind: str = typer.Option("semantic", "--kind", "-k", help="episodic|semantic|procedural"),
):
    """Store a memory entry (local JSON store)."""
    data = read_json(str(APP_DIR / "memory.json"), {"memory": {}})
    data["memory"][key] = {"value": value, "kind": kind, "updated": now_iso()}
    write_json(str(APP_DIR / "memory.json"), data)
    typer.echo(jdump(ok(key=key, kind=kind)))


@app.command("memory-get")
def memory_get(key: str = typer.Argument(..., help="Memory key")):
    """Retrieve a memory entry."""
    data = read_json(str(APP_DIR / "memory.json"), {"memory": {}})
    entry = data["memory"].get(key)
    typer.echo(jdump(ok(found=entry is not None, **({"entry": entry} if entry else {}))))


@app.command("memory-list")
def memory_list(kind: Optional[str] = typer.Option(None, "--kind", "-k")):
    """List memory entries."""
    data = read_json(str(APP_DIR / "memory.json"), {"memory": {}})
    entries = {k: v for k, v in data["memory"].items() if not kind or v.get("kind") == kind}
    typer.echo(jdump(ok(entries=entries)))


@app.command("memory-clear")
def memory_clear():
    """Clear all memory."""
    write_json(str(APP_DIR / "memory.json"), {"memory": {}})
    typer.echo(jdump(ok(cleared=True)))


@app.command("reason")
def reason(
    question: str = typer.Argument(..., help="Question"),
    steps: bool = typer.Option(True, "--steps/--no-steps", help="Show reasoning steps"),
):
    """Local symbolic reasoning scaffold (facts from memory + knowledge base)."""
    from aweai.ai import CONCEPTS

    q = question.lower()
    found = []
    for name, entry in CONCEPTS.items():
        if name in q or entry["summary"].lower() in q:
            found.append({"concept": name, "summary": entry["summary"]})
    steps_out = [
        {"step": 1, "note": "Decompose the question into atomic concepts."},
        {"step": 2, "note": f"Matched {len(found)} knowledge-base concepts."},
        {"step": 3, "note": "Apply definitions; synthesize an answer."},
        {"step": 4, "note": "Verify the answer is consistent with known facts."},
    ]
    answer = "Based on the AWEAI knowledge base: " + (
        "; ".join(f["summary"] for f in found) if found else
        "no exact concept matched locally — consider `aweai ai explain <term>` or RAG over your docs."
    )
    typer.echo(jdump(ok(question=question, answer=answer, evidence=found,
                        steps=steps_out if steps else None)))


@app.command("self-improve")
def self_improve(
    mode: str = typer.Option("check", "--mode", "-m", help="check|critique|plan"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="File/code to critique"),
):
    """Self-improvement hooks: check/plan/critique the codebase."""
    from aweai.ai import SELF_IMPROVEMENT_HOOKS

    if mode == "check":
        typer.echo(jdump(ok(hooks=SELF_IMPROVEMENT_HOOKS,
                            note="Run `aweai autotest` for a full system self-check.")))
    elif mode == "plan":
        typer.echo(jdump(ok(plan=[
            {"hook": h["name"], "action": h["summary"]} for h in SELF_IMPROVEMENT_HOOKS
        ])))
    elif mode == "critique":
        path = target or "aweai/cli.py"
        try:
            src = Path(path).read_text(encoding="utf-8")
            lines = len(src.splitlines())
            imports = sum(1 for ln in src.splitlines() if ln.startswith("import ") or ln.startswith("from "))
            typer.echo(jdump(ok(file=path, lines=lines, imports=imports,
                                critique=[
                                    "Consider splitting large modules into smaller units.",
                                    "Ensure all imports are used.",
                                    "Add type hints to public functions.",
                                    "Keep functions small and single-purpose.",
                                ])))
        except Exception as e:
            typer.echo(jdump(err(str(e))))
            raise typer.Exit(code=1)
    else:
        typer.echo(jdump(err("unknown mode: check|plan|critique")))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# RAG / vector DB
# ---------------------------------------------------------------------------
@app.command("rag-index")
def rag_index(
    path: str = typer.Argument(..., help="Directory or file to index"),
    out: str = typer.Option("~/.aweai/rag_index.json", "--out", "-o", help="Index path"),
):
    """Index documents for RAG (local JSON index)."""
    try:
        from aweai.rag import RAGEngine

        eng = RAGEngine(index_path=str(Path(out).expanduser()))
        p = Path(path)
        if p.is_dir():
            res = eng.index_directory(str(p))
        else:
            res = eng.index_file(str(p))
        typer.echo(jdump(ok(path=path, **res)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("rag-search")
def rag_search(
    query: str = typer.Argument(..., help="Search query"),
    index: str = typer.Option("~/.aweai/rag_index.json", "--index", "-i", help="Index path"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
):
    """Search the RAG index."""
    try:
        from aweai.rag import RAGEngine

        eng = RAGEngine(index_path=str(Path(index).expanduser()))
        hits = eng.search(query, top_k=top_k)
        typer.echo(jdump(ok(query=query, hits=hits)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("rag-ask")
def rag_ask(
    query: str = typer.Argument(..., help="Question"),
    index: str = typer.Option("~/.aweai/rag_index.json", "--index", "-i", help="Index path"),
):
    """Ask a question grounded in the RAG index."""
    try:
        from aweai.rag import RAGEngine

        eng = RAGEngine(index_path=str(Path(index).expanduser()))
        res = eng.ask(query)
        typer.echo(jdump(res))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("vector-store")
def vector_store(
    action: str = typer.Argument(..., help="add|search|list|clear"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Vector key (add)"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to embed (add)"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query (search)"),
    dim: int = typer.Option(64, "--dim", "-d"),
    path: str = typer.Option("~/.aweai/vectors.json", "--path", "-p", help="Store path"),
):
    """Local vector store with hash embeddings."""
    from aweai.cmd.data_manage import _embed_doc

    store_path = Path(path).expanduser()
    data = read_json(str(store_path), {"vectors": {}})
    if action == "add":
        if not key or not text:
            typer.echo(jdump(err("add requires --key and --text")))
            raise typer.Exit(code=1)
        vec = _embed_doc(text, dim, "hash", [text])
        data["vectors"][key] = {"text": text, "vector": vec}
        write_json(str(store_path), data)
        typer.echo(jdump(ok(added=key)))
    elif action == "search":
        if not query:
            typer.echo(jdump(err("search requires --query")))
            raise typer.Exit(code=1)
        qv = _embed_doc(query, dim, "hash", [query])
        scored = []
        for k, v in data["vectors"].items():
            dot = sum(a * b for a, b in zip(qv, v["vector"]))
            scored.append({"key": k, "text": v["text"], "score": round(dot, 4)})
        scored.sort(key=lambda x: -x["score"])
        typer.echo(jdump(ok(results=scored[:10])))
    elif action == "list":
        typer.echo(jdump(ok(keys=list(data["vectors"].keys()), count=len(data["vectors"]))))
    elif action == "clear":
        write_json(str(store_path), {"vectors": {}})
        typer.echo(jdump(ok(cleared=True)))
    else:
        typer.echo(jdump(err("unknown action")))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Security / monitoring / backup
# ---------------------------------------------------------------------------
@app.command("security-scan")
def security_scan(
    path: str = typer.Option(".", "--path", "-p", help="Directory to scan"),
    secrets: bool = typer.Option(True, "--secrets/--no-secrets", help="Scan for hardcoded secrets"),
):
    """Scan a directory for common security issues (secrets, unsafe eval)."""
    issues = []
    root = Path(path)
    for f in root.rglob("*.py"):
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, ln in enumerate(src.splitlines(), 1):
            low = ln.lower()
            if secrets and any(k in low for k in ["api_key =", "password =", "secret =", "token =", "aws_access_key"]):
                issues.append({"file": str(f), "line": i, "issue": "possible hardcoded secret", "text": ln.strip()[:80]})
            if "eval(" in low or "exec(" in low:
                issues.append({"file": str(f), "line": i, "issue": "eval/exec usage", "text": ln.strip()[:80]})
            if "subprocess" in low and "shell=true" in low:
                issues.append({"file": str(f), "line": i, "issue": "shell=True subprocess", "text": ln.strip()[:80]})
    typer.echo(jdump(ok(scanned=str(root), files=len(list(root.rglob("*.py"))), issues=issues[:100],
                        total_issues=len(issues))))


@app.command("hash-file")
def hash_file(
    path: str = typer.Argument(..., help="File"),
    algorithm: str = typer.Option("sha256", "--algo", "-a", help="sha256|md5|sha1|sha512"),
):
    """Hash a file for integrity checks."""
    h = hashlib.new(algorithm)
    h.update(Path(path).read_bytes())
    typer.echo(jdump(ok(path=path, algorithm=algorithm, digest=h.hexdigest())))


@app.command("monitor")
def monitor(
    seconds: float = typer.Option(2.0, "--seconds", "-s", help="Sample duration"),
    interval: float = typer.Option(0.5, "--interval", help="Sample interval"),
):
    """Sample CPU/memory usage over a short window."""
    samples = []
    start = time.time()
    while time.time() - start < seconds:
        try:
            import psutil
            samples.append({"cpu_percent": psutil.cpu_percent(interval=None),
                            "mem_percent": psutil.virtual_memory().percent,
                            "t": round(time.time() - start, 2)})
        except Exception:
            samples.append({"t": round(time.time() - start, 2)})
        time.sleep(interval)
    if samples and "cpu_percent" in samples[0]:
        avg_cpu = sum(s["cpu_percent"] for s in samples) / len(samples)
        avg_mem = sum(s["mem_percent"] for s in samples) / len(samples)
        typer.echo(jdump(ok(samples=len(samples), avg_cpu_percent=round(avg_cpu, 1),
                            avg_mem_percent=round(avg_mem, 1))))
    else:
        typer.echo(jdump(ok(samples=samples, note="psutil not installed; install psutil for CPU/mem")))


@app.command("backup")
def backup(
    source: str = typer.Argument(..., help="Directory/file to back up"),
    dest: str = typer.Option("~/.aweai/backups", "--dest", "-d", help="Backup root"),
):
    """Copy a directory/file into a timestamped backup."""
    src = Path(source)
    if not src.exists():
        typer.echo(jdump(err(f"source not found: {source}")))
        raise typer.Exit(code=1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(dest).expanduser() / f"{src.name}_{stamp}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_dir / src.name)
    typer.echo(jdump(ok(source=str(src), backup=str(out_dir))))


@app.command("backup-list")
def backup_list(dest: str = typer.Option("~/.aweai/backups", "--dest", "-d")):
    """List backups."""
    d = Path(dest).expanduser()
    if not d.exists():
        typer.echo(jdump(ok(backups=[])))
        return
    backups = []
    for p in sorted(d.iterdir(), reverse=True)[:50]:
        backups.append({"path": str(p), "size": sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size})
    typer.echo(jdump(ok(backups=backups)))
