"""AWEAI core tools — system, file, process, network and information utilities.

Each tool has a unique purpose. All tools are dependency-light and safe to
run in any environment (localhost, container, cloud, phone browser backend).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

@tool("sys_info", "core", "Report operating system, machine and Python runtime details")
def sys_info() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


@tool("env_vars", "core", "List environment variables (optionally filtered by prefix)")
def env_vars(prefix: str = "") -> Dict[str, Any]:
    return {"env": {k: v for k, v in os.environ.items() if k.startswith(prefix)}}


@tool("get_env", "core", "Read a single environment variable")
def get_env(name: str) -> Dict[str, Any]:
    return {"name": name, "value": os.environ.get(name, "")}


@tool("set_env", "core", "Set an environment variable for the current process")
def set_env(name: str, value: str) -> Dict[str, Any]:
    os.environ[name] = value
    return {"name": name, "value": value, "set": True}


@tool("uuid_gen", "core", "Generate a random UUID (v4)")
def uuid_gen() -> Dict[str, Any]:
    return {"uuid": str(uuid.uuid4())}


@tool("now", "core", "Current timestamp in ISO format with timezone offset")
def now(utc: bool = False) -> Dict[str, Any]:
    if utc:
        return {"now": _dt.datetime.now(_dt.timezone.utc).isoformat(), "utc": True}
    return {"now": _dt.datetime.now().astimezone().isoformat(), "utc": False}


@tool("uptime", "core", "System uptime in seconds and human readable form")
def uptime() -> Dict[str, Any]:
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            secs = float(f.read().split()[0])
        return {"seconds": secs, "human": f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m {int(secs % 60)}s"}
    except Exception:
        return {"seconds": 0, "human": "unknown"}


@tool("hostname", "core", "Show the machine hostname")
def hostname() -> Dict[str, Any]:
    return {"hostname": socket.gethostname()}


@tool("ip_local", "core", "Show local IP addresses (IPv4 + IPv6)")
def ip_local() -> Dict[str, Any]:
    ips = []
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ips.append(info[4][0])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return {"ips": sorted(set(ips))}


@tool("disk_usage", "core", "Show disk usage for a path (total, used, free)")
def disk_usage(path: str = ".") -> Dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "percent_used": round(usage.used / usage.total * 100, 2) if usage.total else 0,
    }


@tool("cpu_count", "core", "Number of logical CPU cores")
def cpu_count() -> Dict[str, Any]:
    return {"logical": os.cpu_count() or 0}


@tool("load_avg", "core", "System load average (1/5/15 minutes) when available")
def load_avg() -> Dict[str, Any]:
    try:
        return {"load": list(os.getloadavg())}
    except Exception:
        return {"load": None}


@tool("memory_info", "core", "System memory info (total/available from /proc/meminfo or psutil-free)")
def memory_info() -> Dict[str, Any]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            data = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    data[parts[0].strip()] = parts[1].strip()
        return {
            "total_kb": data.get("MemTotal"),
            "available_kb": data.get("MemAvailable"),
            "free_kb": data.get("MemFree"),
        }
    except Exception:
        return {"total_kb": None}


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

@tool("file_list", "core", "List directory contents (names, sizes, mtimes)")
def file_list(path: str = ".", recursive: bool = False) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_file():
        return {"path": str(p), "files": [{"name": p.name, "size": p.stat().st_size}]}
    entries = []
    for child in sorted(p.iterdir()):
        entries.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
                "mtime": child.stat().st_mtime,
            }
        )
        if recursive and child.is_dir():
            entries.extend(file_list(str(child), recursive=True).get("files", []))
    return {"path": str(p), "files": entries}


@tool("file_read", "core", "Read a text file (optionally first N lines)")
def file_read(path: str, lines: Optional[int] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if lines is not None:
        text = "\n".join(text.splitlines()[:lines])
    return {"path": str(p), "lines": len(text.splitlines()), "content": text}


@tool("file_write", "core", "Write text content to a file (creates parent dirs)")
def file_write(path: str, content: str) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"path": str(p), "bytes": p.stat().st_size, "written": True}


@tool("file_append", "core", "Append text to a file (creates if missing)")
def file_append(path: str, content: str) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    return {"path": str(p), "appended": True}


@tool("file_copy", "core", "Copy a file or directory tree to a destination")
def file_copy(src: str, dst: str) -> Dict[str, Any]:
    s, d = Path(src), Path(dst)
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    return {"src": src, "dst": dst, "copied": True}


@tool("file_move", "core", "Move/rename a file or directory")
def file_move(src: str, dst: str) -> Dict[str, Any]:
    s, d = Path(src), Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return {"src": src, "dst": dst, "moved": True}


@tool("file_delete", "core", "Delete a file or empty directory tree")
def file_delete(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()
    else:
        raise FileNotFoundError(path)
    return {"path": path, "deleted": True}


@tool("file_info", "core", "Show file metadata (size, mtime, mode, type)")
def file_info(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    st = p.stat()
    return {
        "path": str(p),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "mode": oct(st.st_mode),
        "type": "dir" if p.is_dir() else "file",
        "absolute": str(p.resolve()),
    }


@tool("file_find", "core", "Find files by name glob under a directory")
def file_find(pattern: str, root: str = ".") -> Dict[str, Any]:
    matches = [str(p) for p in Path(root).rglob(pattern)]
    return {"pattern": pattern, "root": root, "matches": matches, "count": len(matches)}


@tool("file_touch", "core", "Create an empty file or update its mtime")
def file_touch(path: str) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        os.utime(p, None)
    else:
        p.touch()
    return {"path": str(p), "touched": True}


@tool("file_tree", "core", "Print directory tree as indented text")
def file_tree(root: str = ".", depth: int = 3) -> Dict[str, Any]:
    lines: List[str] = []

    def walk(p: Path, level: int) -> None:
        if level > depth:
            return
        lines.append("  " * level + p.name + ("/" if p.is_dir() else ""))
        if p.is_dir():
            for child in sorted(p.iterdir()):
                walk(child, level + 1)

    walk(Path(root), 0)
    return {"root": root, "tree": "\n".join(lines)}


@tool("file_checksum", "core", "Compute SHA-256 checksum of a file (streamed)")
def file_checksum(path: str) -> Dict[str, Any]:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {"path": path, "sha256": h.hexdigest()}


@tool("file_tail", "core", "Show the last N lines of a file (like tail)")
def file_tail(path: str, lines: int = 10) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    content = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"path": str(p), "tail": content[-lines:]}


@tool("file_head", "core", "Show the first N lines of a file (like head)")
def file_head(path: str, lines: int = 10) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    content = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"path": str(p), "head": content[:lines]}


@tool("file_count", "core", "Count lines, words and characters in a text file")
def file_count(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        "path": path,
        "lines": len(text.splitlines()),
        "words": len(text.split()),
        "chars": len(text),
    }


@tool("file_ext", "core", "Extract and list file extensions under a directory")
def file_ext(root: str = ".") -> Dict[str, Any]:
    from collections import Counter

    counts: Counter = Counter()
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix:
            counts[p.suffix.lower()] += 1
    return {"root": root, "extensions": dict(counts.most_common())}


# ---------------------------------------------------------------------------
# Process tools
# ---------------------------------------------------------------------------

@tool("process_list", "core", "List running processes (via ps, best effort)")
def process_list(filter: str = "") -> Dict[str, Any]:
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {"processes": [], "error": "ps not available"}
    lines = out.splitlines()
    header = lines[0] if lines else ""
    rows = []
    for line in lines[1:]:
        if filter and filter not in line:
            continue
        rows.append(line)
    return {"header": header, "count": len(rows), "processes": rows[:200]}


@tool("process_ps", "core", "Snapshot of process info for a PID (name, cpu, mem, cmd)")
def process_ps(pid: int) -> Dict[str, Any]:
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "pid,pcpu,pmem,rss,etime,comm,args"],
                             capture_output=True, text=True, timeout=10).stdout
        return {"pid": pid, "ps": out.strip()}
    except Exception as e:
        return {"pid": pid, "error": str(e)}


@tool("process_kill", "core", "Terminate a process by PID (SIGTERM first, SIGKILL if force)")
def process_kill(pid: int, force: bool = False) -> Dict[str, Any]:
    import signal

    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.kill(pid, sig)
        return {"pid": pid, "signal": sig, "sent": True}
    except ProcessLookupError:
        return {"pid": pid, "error": "no such process"}
    except PermissionError:
        return {"pid": pid, "error": "permission denied"}


@tool("process_run", "core", "Run a shell command and capture stdout/stderr/return code")
def process_run(command: str, timeout: int = 30) -> Dict[str, Any]:
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "returncode": res.returncode,
            "stdout": res.stdout[-4000:],
            "stderr": res.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "error": f"timed out after {timeout}s"}
    except Exception as e:
        return {"command": command, "error": str(e)}


@tool("process_pidof", "core", "Find PIDs by process name")
def process_pidof(name: str) -> Dict[str, Any]:
    try:
        out = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True, timeout=10).stdout
        pids = [int(x) for x in out.split()]
        return {"name": name, "pids": pids}
    except Exception:
        return {"name": name, "pids": []}


# ---------------------------------------------------------------------------
# Network / connectivity
# ---------------------------------------------------------------------------

@tool("ping", "core", "Ping a host and report round-trip time (ICMP via system ping)")
def ping(host: str, count: int = 4) -> Dict[str, Any]:
    try:
        out = subprocess.run(["ping", "-c", str(count), host], capture_output=True, text=True, timeout=30).stdout
        return {"host": host, "output": out[-1500:]}
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("http_get", "core", "Fetch a URL and return status + first bytes of body")
def http_get(url: str, timeout: int = 15) -> Dict[str, Any]:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(4000).decode("utf-8", errors="replace")
            return {"url": url, "status": r.status, "body": body}
    except Exception as e:
        return {"url": url, "error": str(e)}


@tool("dns_lookup", "core", "Resolve a hostname to IP addresses (A records)")
def dns_lookup(host: str) -> Dict[str, Any]:
    try:
        return {"host": host, "ips": sorted({i[4][0] for i in socket.getaddrinfo(host, None, socket.AF_INET)})}
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("port_check", "core", "Check whether a TCP port is open on a host")
def port_check(host: str, port: int, timeout: int = 3) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return {"host": host, "port": port, "open": True}
    except Exception:
        return {"host": host, "port": port, "open": False}
    finally:
        s.close()


@tool("tcp_connect", "core", "Open a raw TCP connection and send/receive a short payload")
def tcp_connect(host: str, port: int, payload: str = "", timeout: int = 5) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        if payload:
            s.sendall(payload.encode("utf-8"))
        data = s.recv(2048).decode("utf-8", errors="replace")
        return {"host": host, "port": port, "response": data}
    except Exception as e:
        return {"host": host, "port": port, "error": str(e)}
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Temp / workspace
# ---------------------------------------------------------------------------

@tool("temp_dir", "core", "Create a temporary directory and return its path")
def temp_dir(prefix: str = "aweai_") -> Dict[str, Any]:
    return {"path": tempfile.mkdtemp(prefix=prefix)}


@tool("temp_file", "core", "Create a temporary file and return its path")
def temp_file(prefix: str = "aweai_", suffix: str = ".tmp") -> Dict[str, Any]:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return {"path": path}


@tool("cwd", "core", "Show current working directory")
def cwd() -> Dict[str, Any]:
    return {"cwd": os.getcwd()}


@tool("ls_home", "core", "List files in the user home directory")
def ls_home() -> Dict[str, Any]:
    return file_list(str(Path.home()))


@tool("sleep_ms", "core", "Sleep for a number of milliseconds (delays execution)")
def sleep_ms(ms: int = 100) -> Dict[str, Any]:
    time.sleep(ms / 1000.0)
    return {"slept_ms": ms}


@tool("json_parse", "core", "Parse a JSON string and return the object")
def json_parse(text: str) -> Dict[str, Any]:
    return {"parsed": json.loads(text)}


@tool("json_dumps", "core", "Serialize an object to a JSON string")
def json_dumps(obj: str) -> Dict[str, Any]:
    try:
        data = json.loads(obj)
    except json.JSONDecodeError:
        data = obj
    return {"json": json.dumps(data, ensure_ascii=False, indent=2)}


@tool("echo", "core", "Echo text back (useful for pipelines and testing)")
def echo(text: str = "") -> Dict[str, Any]:
    return {"echo": text}


@tool("random_number", "core", "Generate a random integer in a range")
def random_number(lo: int = 0, hi: int = 100) -> Dict[str, Any]:
    import random

    return {"value": random.randint(lo, hi)}


@tool("random_string", "core", "Generate a random alphanumeric string")
def random_string(length: int = 12) -> Dict[str, Any]:
    import random
    import string

    chars = string.ascii_letters + string.digits
    return {"value": "".join(random.choice(chars) for _ in range(length))}


@tool("path_join", "core", "Join path components safely")
def path_join(*parts: str) -> Dict[str, Any]:
    return {"path": str(Path(*parts))}


@tool("path_abs", "core", "Return the absolute path of a relative path")
def path_abs(path: str) -> Dict[str, Any]:
    return {"absolute": str(Path(path).resolve())}


@tool("path_parent", "core", "Return the parent directory of a path")
def path_parent(path: str) -> Dict[str, Any]:
    return {"parent": str(Path(path).parent)}


@tool("path_basename", "core", "Return the basename of a path")
def path_basename(path: str) -> Dict[str, Any]:
    return {"basename": Path(path).name}


@tool("time_iso", "core", "Current time as ISO 8601 string")
def time_iso() -> Dict[str, Any]:
    return {"iso": _dt.datetime.now().astimezone().isoformat()}


@tool("date_today", "core", "Today's date as YYYY-MM-DD")
def date_today() -> Dict[str, Any]:
    return {"date": _dt.date.today().isoformat()}


@tool("uname", "core", "System uname-style information")
def uname() -> Dict[str, Any]:
    return {
        "sysname": platform.system(),
        "nodename": socket.gethostname(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


@tool("arch", "core", "CPU architecture")
def arch() -> Dict[str, Any]:
    return {"arch": platform.machine()}


@tool("shell_quote", "core", "Shell-quote a string for safe command usage")
def shell_quote(text: str) -> Dict[str, Any]:
    import shlex

    return {"quoted": shlex.quote(text)}


__all__ = [t["name"] for t in []]
