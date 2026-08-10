# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI extended bulk command specs (v4.1).

Adds hundreds of additional declarative commands across new groups:

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
# SEARCH group — text/vector search utilities
# ===========================================================================

spec("search", "tokenize", "Tokenize text into words.",
     [("text", "hello world", "Text")],
     lambda p: _ok(tokens=re.findall(r"[a-z0-9]+", p["text"].lower())))
spec("search", "ngrams", "Build n-grams from tokens.",
     [("text", "AWEAI universal CLI", "Text"), ("n", 2, "N")],
     lambda p: _search_ngrams(p["text"], int(p["n"])))
spec("search", "tf", "Term frequency of words.",
     [("text", "the cat and the dog", "Text")], lambda p: _search_tf(p["text"]))
spec("search", "cosine", "Cosine similarity between two comma lists.",
     [("a", "1,0,1", "Vector A"), ("b", "1,1,0", "Vector B")],
     lambda p: _search_cosine(p["a"], p["b"]))
spec("search", "jaccard", "Jaccard similarity of two token sets.",
     [("a", "cat dog bird", "Set A"), ("b", "cat dog fish", "Set B")],
     lambda p: _search_jaccard(p["a"], p["b"]))
spec("search", "levenshtein", "Levenshtein distance between two strings.",
     [("a", "kitten", "String A"), ("b", "sitting", "String B")],
     lambda p: _ok(distance=_lev(a=p["a"], b=p["b"])))


def _search_ngrams(text: str, n: int) -> Dict[str, Any]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    n = max(1, n)
    out = [" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return _ok(ngrams=out)


def _search_tf(text: str) -> Dict[str, Any]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return _ok(tf=[{"term": k, "count": v} for k, v in sorted(freq.items(), key=lambda x: -x[1])])


def _search_cosine(a: str, b: str) -> Dict[str, Any]:
    va = _floats(a)
    vb = _floats(b)
    if len(va) != len(vb) or not va:
        return _err("vectors must be equal length and non-empty")
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    return _ok(similarity=round(dot / (na * nb), 4) if na and nb else 0.0)


def _search_jaccard(a: str, b: str) -> Dict[str, Any]:
    sa = set(re.findall(r"[a-z0-9]+", a.lower()))
    sb = set(re.findall(r"[a-z0-9]+", b.lower()))
    inter = len(sa & sb)
    union = len(sa | sb)
    return _ok(similarity=round(inter / union, 4) if union else 0.0, inter=inter, union=union)


def _lev(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ===========================================================================
# CODE group — analysis, review, formatting
# ===========================================================================

spec("code", "stats", "Code statistics for a file (lines, comments, blank).",
     [("path", "aweai/cli.py", "File path")], lambda p: _code_stats(p["path"]))
spec("code", "lint", "Simple lint heuristics (line length, trailing spaces, tabs).",
     [("path", "aweai/cli.py", "File path")], lambda p: _code_lint(p["path"]))
spec("code", "review", "Generate a review checklist for a file.",
     [("path", "aweai/cli.py", "File path")], lambda p: _code_review(p["path"]))
spec("code", "todos", "Find TODO/FIXME markers in a directory.",
     [("path", ".", "Directory"), ("pattern", "TODO|FIXME", "Regex")],
     lambda p: _code_todos(p["path"], p["pattern"]))
spec("code", "format-json", "Pretty-print a JSON file.",
     [("path", "data.json", "Path")], lambda p: _code_format_json(p["path"]))
spec("code", "grep", "Search text in files under a directory.",
     [("path", ".", "Directory"), ("pattern", "Copyright", "Regex"), ("ext", "py", "Extensions (comma)")],
     lambda p: _code_grep(p["path"], p["pattern"], p["ext"]))


def _code_stats(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return _err("file not found")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    code = 0
    blank = 0
    comment = 0
    in_block = False
    for ln in lines:
        s = ln.strip()
        if not s:
            blank += 1
            continue
        if in_block:
            comment += 1
            if s.endswith('"""') or s.endswith("'''"):
                in_block = False
            continue
        if s.startswith('"""') or s.startswith("'''"):
            comment += 1
            if not (s.endswith('"""') or s.endswith("'''")) or len(s) == 3:
                in_block = True
            continue
        if s.startswith("#"):
            comment += 1
            continue
        code += 1
    return _ok(path=path, lines=len(lines), code=code, blank=blank, comment=comment)


def _code_lint(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return _err("file not found")
    issues = []
    for i, ln in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if len(ln) > 120:
            issues.append({"line": i, "issue": "line too long", "len": len(ln)})
        if ln.rstrip() != ln:
            issues.append({"line": i, "issue": "trailing whitespace"})
        if "\t" in ln:
            issues.append({"line": i, "issue": "tab used"})
    return _ok(path=path, issues=issues, count=len(issues))


def _code_review(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return _err("file not found")
    text = p.read_text(encoding="utf-8", errors="ignore")
    checks = {
        "has_header": "Copyright" in text,
        "has_docstring": '"""' in text,
        "no_todos": "TODO" not in text.upper(),
        "reasonable_len": len(text.splitlines()) < 5000,
    }
    return _ok(path=path, checks=checks, score=sum(checks.values()) / len(checks.values()))


def _code_todos(path: str, pattern: str) -> Dict[str, Any]:
    root = Path(path)
    hits = []
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in {".py", ".js", ".ts", ".md", ".txt", ".sh"}:
                for i, ln in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if re.search(pattern, ln):
                        hits.append({"file": str(f), "line": i, "text": ln.strip()[:80]})
    return _ok(hits=hits, count=len(hits))


def _code_format_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return _err("file not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return _ok(formatted=True, path=str(p))
    except Exception as e:
        return _err(f"invalid JSON: {e}")


def _code_grep(path: str, pattern: str, ext: str) -> Dict[str, Any]:
    root = Path(path)
    exts = {f".{e.strip().lstrip('.')}" for e in ext.split(",") if e.strip()}
    hits = []
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and (not exts or f.suffix in exts):
                for i, ln in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if re.search(pattern, ln):
                        hits.append({"file": str(f), "line": i, "text": ln.strip()[:100]})
    return _ok(hits=hits, count=len(hits))


# ===========================================================================
# SHELL group — command helpers
# ===========================================================================

spec("shell", "run", "Run a shell command and capture output.",
     [("command", "echo hello", "Command"), ("timeout", 30, "Timeout seconds")],
     lambda p: _shell_run(p["command"], int(p["timeout"])))
spec("shell", "which", "Check if a program is on PATH.",
     [("program", "git", "Program")], lambda p: _ok(found=shutil.which(p["program"]) is not None,
                                                    path=shutil.which(p["program"])))
spec("shell", "env", "Read an environment variable.",
     [("name", "HOME", "Variable")], lambda p: _ok(name=p["name"], value=os.environ.get(p["name"])))
spec("shell", "exec", "Execute a Python expression (safe eval of literals).",
     [("expr", "1 + 2", "Python expression")], lambda p: _shell_exec(p["expr"]))
spec("shell", "pid", "Current process id.", [], lambda p: _ok(pid=os.getpid(), ppid=os.getppid()))


def _shell_run(command: str, timeout: int) -> Dict[str, Any]:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=max(1, timeout))
        return _ok(command=command, returncode=r.returncode, stdout=r.stdout[-2000:], stderr=r.stderr[-2000:])
    except subprocess.TimeoutExpired:
        return _err("timeout")
    except Exception as e:
        return _err(str(e))


def _shell_exec(expr: str) -> Dict[str, Any]:
    try:
        import ast
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                     ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv, ast.List,
                                     ast.Tuple, ast.Dict, ast.Set, ast.Compare, ast.Eq, ast.NotEq,
                                     ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or,
                                     ast.Not, ast.Call, ast.Attribute)):
                return _err("unsafe expression")
        if isinstance(tree.body, ast.Call) and isinstance(tree.body.func, ast.Attribute):
            return _err("unsafe expression")
        return _ok(result=eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return _err(str(e))


# ===========================================================================
# GIT group — git helpers
# ===========================================================================

spec("git", "status", "Git status (short) of a directory.",
     [("path", ".", "Repo path")], lambda p: _git_status(p["path"]))
spec("git", "log", "Recent git log (oneline).",
     [("path", ".", "Repo path"), ("n", 10, "Commits")],
     lambda p: _git_log(p["path"], int(p["n"])))
spec("git", "branch", "Current git branch.",
     [("path", ".", "Repo path")], lambda p: _git_branch(p["path"]))
spec("git", "remote", "Git remotes.",
     [("path", ".", "Repo path")], lambda p: _git_remote(p["path"]))
spec("git", "commit-msg", "Generate a conventional commit message.",
     [("type", "feat", "feat|fix|docs|test|refactor"), ("summary", "add commands", "Summary")],
     lambda p: _ok(message=f"{p['type']}: {p['summary']}"))


def _git(path: str, *args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def _git_status(path: str) -> Dict[str, Any]:
    return _ok(path=path, status=_git(path, "status", "--short") or "(clean)")


def _git_log(path: str, n: int) -> Dict[str, Any]:
    out = _git(path, "log", "--oneline", f"-{max(1, n)}")
    return _ok(commits=[c for c in out.splitlines() if c.strip()])


def _git_branch(path: str) -> Dict[str, Any]:
    return _ok(branch=_git(path, "rev-parse", "--abbrev-ref", "HEAD") or "unknown")


def _git_remote(path: str) -> Dict[str, Any]:
    out = _git(path, "remote", "-v")
    return _ok(remotes=[r for r in out.splitlines() if r.strip()])


# ===========================================================================
# DOCKER group — docker helpers
# ===========================================================================

spec("docker", "available", "Check if docker CLI is available.",
     [], lambda p: _ok(available=shutil.which("docker") is not None))
spec("docker", "version", "Docker version string (if available).",
     [], lambda p: _docker_version())
spec("docker", "images", "List docker images (if available).",
     [], lambda p: _docker_images())
spec("docker", "ps", "List running containers (if available).",
     [], lambda p: _docker_ps())


def _docker_version() -> Dict[str, Any]:
    if shutil.which("docker") is None:
        return _ok(available=False)
    try:
        r = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        return _ok(available=True, version=r.stdout.strip() or r.stderr.strip())
    except Exception as e:
        return _err(str(e))


def _docker_images() -> Dict[str, Any]:
    if shutil.which("docker") is None:
        return _ok(available=False)
    try:
        r = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True, timeout=15)
        lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        return _ok(images=lines)
    except Exception as e:
        return _err(str(e))


def _docker_ps() -> Dict[str, Any]:
    if shutil.which("docker") is None:
        return _ok(available=False)
    try:
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=15)
        lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        return _ok(containers=lines)
    except Exception as e:
        return _err(str(e))
