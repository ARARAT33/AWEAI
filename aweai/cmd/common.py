# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Shared helpers for the AWEAI command groups (aweai.cmd)."""

from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_DIR = Path(os.environ.get("AWEAI_HOME", str(Path.home() / ".aweai")))


def data_dir() -> Path:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    return APP_DIR


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def ok(**kw: Any) -> Dict[str, Any]:
    return {"ok": True, **kw}


def err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


def http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "AWEAI-CLI/4.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def http_get_json(url: str, timeout: int = 20) -> Any:
    return json.loads(http_get(url, timeout=timeout))


def online(host: str = "api.github.com", port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=4):
            return True
    except Exception:
        return False


def read_json(path: str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: str, obj: Any) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return ok(path=str(p))


def read_text(path: str, default: str = "") -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else default


def write_text(path: str, text: str) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return ok(path=str(p), bytes=len(text.encode("utf-8")))


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd: str, timeout: int = 60) -> str:
    import subprocess

    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""


def model_path(name: str) -> Path:
    return APP_DIR / "models" / f"{name}.json"


def list_models_local() -> List[Dict[str, Any]]:
    d = APP_DIR / "models"
    out = []
    if d.exists():
        for f in sorted(d.glob("*.json")):
            meta = read_json(str(f), {})
            out.append({"name": f.stem, "size": f.stat().st_size, "meta": meta.get("meta", {})})
    return out


def git_repo_root(path: str = ".") -> Optional[Path]:
    p = Path(path).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None
