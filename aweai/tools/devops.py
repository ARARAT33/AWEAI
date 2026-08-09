"""AWEAI devops tools — git, docker, CI, deploy and infrastructure helpers.

Each tool has a unique purpose. Tools are shell-friendly and dependency-light.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool


def _sh(cmd: str, timeout: int = 30) -> Dict[str, Any]:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "returncode": res.returncode,
            "stdout": res.stdout[-3000:],
            "stderr": res.stderr[-1500:],
        }
    except subprocess.TimeoutExpired:
        return {"command": cmd, "error": f"timed out after {timeout}s"}
    except Exception as e:
        return {"command": cmd, "error": str(e)}


def _in_repo(path: str = ".") -> Path:
    root = Path(path).resolve()
    if not (root / ".git").exists():
        raise ValueError(f"Not a git repository: {path}")
    return root


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

@tool("git_status", "devops", "Show git status (porcelain) of a repository")
def git_status(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git status --porcelain=v1 -b")


@tool("git_log", "devops", "Show recent git commit log")
def git_log(path: str = ".", count: int = 20) -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git log --oneline -n {count}")


@tool("git_branch", "devops", "List git branches (current marked)")
def git_branch(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git branch -a")


@tool("git_current_branch", "devops", "Show the current git branch name")
def git_current_branch(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git rev-parse --abbrev-ref HEAD")


@tool("git_commit", "devops", "Commit staged changes with a message (no push)")
def git_commit(path: str, message: str) -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f'cd {path} && git add -A && git commit -m "{message}"')


@tool("git_commit_all", "devops", "Stage all and commit with a message")
def git_commit_all(path: str, message: str) -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f'cd {path} && git add -A && git commit -m "{message}"')


@tool("git_push", "devops", "Push commits to the remote (requires credentials)")
def git_push(path: str = ".", remote: str = "origin", branch: str = "") -> Dict[str, Any]:
    _in_repo(path)
    if branch:
        return _sh(f"cd {path} && git push {remote} {branch}")
    return _sh(f"cd {path} && git push {remote}")


@tool("git_pull", "devops", "Pull latest changes from the remote")
def git_pull(path: str = ".", remote: str = "origin", branch: str = "main") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git pull {remote} {branch}")


@tool("git_clone", "devops", "Clone a git repository into a local directory")
def git_clone(url: str, dest: str = "") -> Dict[str, Any]:
    cmd = f"git clone {url}" + (f" {dest}" if dest else "")
    return _sh(cmd, timeout=120)


@tool("git_remote", "devops", "List git remotes")
def git_remote(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git remote -v")


@tool("git_diff", "devops", "Show working-tree diff (stat)")
def git_diff(path: str = ".", stat: bool = True) -> Dict[str, Any]:
    _in_repo(path)
    flag = "--stat" if stat else ""
    return _sh(f"cd {path} && git diff {flag}")


@tool("git_tag", "devops", "Create a git tag")
def git_tag(path: str, tag: str, message: str = "") -> Dict[str, Any]:
    _in_repo(path)
    if message:
        return _sh(f'cd {path} && git tag -a {tag} -m "{message}"')
    return _sh(f"cd {path} && git tag {tag}")


@tool("git_tags", "devops", "List git tags")
def git_tags(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git tag -l")


@tool("git_checkout", "devops", "Checkout a branch or commit")
def git_checkout(path: str, ref: str) -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git checkout {ref}")


@tool("git_stash", "devops", "Stash working changes")
def git_stash(path: str = ".", message: str = "") -> Dict[str, Any]:
    _in_repo(path)
    if message:
        return _sh(f'cd {path} && git stash push -m "{message}"')
    return _sh(f"cd {path} && git stash")


@tool("git_reset", "devops", "Reset working tree to HEAD (soft keeps changes)")
def git_reset(path: str = ".", hard: bool = False) -> Dict[str, Any]:
    _in_repo(path)
    flag = "--hard" if hard else "--soft"
    return _sh(f"cd {path} && git reset {flag} HEAD")


@tool("git_count", "devops", "Count commits on the current branch")
def git_count(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git rev-list --count HEAD")


@tool("git_last_sha", "devops", "Show the last commit SHA (short)")
def git_last_sha(path: str = ".") -> Dict[str, Any]:
    _in_repo(path)
    return _sh(f"cd {path} && git rev-parse --short HEAD")


# ---------------------------------------------------------------------------
# Python / packaging
# ---------------------------------------------------------------------------

@tool("pip_list", "devops", "List installed Python packages")
def pip_list() -> Dict[str, Any]:
    return _sh("pip list --format=freeze 2>/dev/null || pip3 list --format=freeze", timeout=60)


@tool("pip_install", "devops", "Install Python packages with pip")
def pip_install(package: str, upgrade: bool = False) -> Dict[str, Any]:
    flag = " --upgrade" if upgrade else ""
    return _sh(f"pip install {package}{flag}", timeout=300)


@tool("pip_show", "devops", "Show package info (version, location, requires)")
def pip_show(package: str) -> Dict[str, Any]:
    return _sh(f"pip show {package}")


@tool("python_version", "devops", "Show Python interpreter version")
def python_version() -> Dict[str, Any]:
    import sys

    return {"version": sys.version.split()[0], "executable": sys.executable}


@tool("pip_outdated", "devops", "List outdated Python packages")
def pip_outdated() -> Dict[str, Any]:
    return _sh("pip list --outdated --format=columns", timeout=120)


@tool("freeze_reqs", "devops", "Dump installed packages to a requirements file")
def freeze_reqs(path: str = "requirements.txt") -> Dict[str, Any]:
    res = _sh("pip freeze", timeout=60)
    if "error" in res:
        return res
    Path(path).write_text(res.get("stdout", ""), encoding="utf-8")
    return {"path": path, "written": True, "packages": len(res.get("stdout", "").splitlines())}


@tool("venv_create", "devops", "Create a Python virtual environment")
def venv_create(path: str = ".venv") -> Dict[str, Any]:
    return _sh(f"python -m venv {path}", timeout=120)


@tool("venv_activate_check", "devops", "Check whether a virtual environment is active")
def venv_activate_check() -> Dict[str, Any]:
    return {"active": bool(os.environ.get("VIRTUAL_ENV")), "venv": os.environ.get("VIRTUAL_ENV", "")}


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

@tool("docker_version", "devops", "Show Docker client/server version")
def docker_version() -> Dict[str, Any]:
    return _sh("docker --version && docker info --format '{{.ServerVersion}}' 2>/dev/null", timeout=30)


@tool("docker_ps", "devops", "List running Docker containers")
def docker_ps(all: bool = False) -> Dict[str, Any]:
    flag = " -a" if all else ""
    return _sh(f"docker ps{flag}", timeout=30)


@tool("docker_images", "devops", "List Docker images")
def docker_images() -> Dict[str, Any]:
    return _sh("docker images", timeout=30)


@tool("docker_build", "devops", "Build a Docker image from a Dockerfile")
def docker_build(path: str = ".", tag: str = "aweai:latest") -> Dict[str, Any]:
    return _sh(f"docker build -t {tag} {path}", timeout=600)


@tool("docker_pull", "devops", "Pull a Docker image")
def docker_pull(image: str) -> Dict[str, Any]:
    return _sh(f"docker pull {image}", timeout=600)


@tool("docker_run", "devops", "Run a Docker container (detached)")
def docker_run(image: str, name: str = "", ports: str = "", command: str = "") -> Dict[str, Any]:
    parts = ["docker", "run", "-d"]
    if name:
        parts += ["--name", name]
    if ports:
        parts += ["-p", ports]
    parts.append(image)
    if command:
        parts.append(command)
    return _sh(" ".join(parts), timeout=120)


@tool("docker_stop", "devops", "Stop a Docker container")
def docker_stop(name_or_id: str) -> Dict[str, Any]:
    return _sh(f"docker stop {name_or_id}", timeout=60)


@tool("docker_rm", "devops", "Remove a Docker container")
def docker_rm(name_or_id: str, force: bool = False) -> Dict[str, Any]:
    flag = " -f" if force else ""
    return _sh(f"docker rm{flag} {name_or_id}", timeout=60)


@tool("docker_logs", "devops", "Show logs of a Docker container")
def docker_logs(name_or_id: str, tail: int = 100) -> Dict[str, Any]:
    return _sh(f"docker logs --tail {tail} {name_or_id}", timeout=30)


@tool("docker_compose_up", "devops", "Bring up docker compose services")
def docker_compose_up(path: str = ".", detached: bool = True) -> Dict[str, Any]:
    flag = " -d" if detached else ""
    return _sh(f"cd {path} && docker compose up{flag}", timeout=600)


@tool("docker_compose_down", "devops", "Tear down docker compose services")
def docker_compose_down(path: str = ".") -> Dict[str, Any]:
    return _sh(f"cd {path} && docker compose down", timeout=120)


@tool("docker_stats", "devops", "Live Docker container resource stats (one-shot)")
def docker_stats() -> Dict[str, Any]:
    return _sh("docker stats --no-stream", timeout=30)


# ---------------------------------------------------------------------------
# CI / build
# ---------------------------------------------------------------------------

@tool("run_tests", "devops", "Run the project test suite (pytest)")
def run_tests(path: str = ".", extra: str = "-q") -> Dict[str, Any]:
    return _sh(f"cd {path} && python -m pytest {extra}", timeout=600)


@tool("run_pytest", "devops", "Run pytest with a specific test path/expression")
def run_pytest(path: str = ".", test_path: str = "tests") -> Dict[str, Any]:
    return _sh(f"cd {path} && python -m pytest {test_path} -q", timeout=600)


@tool("compile_all", "devops", "Byte-compile all Python sources (python -m compileall)")
def compile_all(path: str = ".") -> Dict[str, Any]:
    return _sh(f"cd {path} && python -m compileall -q aweai", timeout=120)


@tool("lint_flake8", "devops", "Run flake8 linting (if installed)")
def lint_flake8(path: str = ".") -> Dict[str, Any]:
    return _sh(f"cd {path} && python -m flake8 aweai --count --statistics 2>/dev/null || echo 'flake8 not installed'", timeout=120)


@tool("lint_pyflakes", "devops", "Run pyflakes linting (if installed)")
def lint_pyflakes(path: str = ".") -> Dict[str, Any]:
    return _sh(f"cd {path} && python -m pyflakes aweai 2>/dev/null || echo 'pyflakes not installed'", timeout=120)


@tool("build_package", "devops", "Build Python distribution artifacts (sdist + wheel)")
def build_package(path: str = ".") -> Dict[str, Any]:
    return _sh(f"cd {path} && python -m build 2>/dev/null || pip wheel . -w dist", timeout=600)


@tool("make_target", "devops", "Run a Makefile target")
def make_target(path: str, target: str = "test") -> Dict[str, Any]:
    return _sh(f"cd {path} && make {target}", timeout=600)


@tool("shell_check", "devops", "Check a shell script for syntax errors (bash -n)")
def shell_check(path: str) -> Dict[str, Any]:
    return _sh(f"bash -n {path}")


@tool("yaml_validate", "devops", "Validate a YAML file (via python yaml if installed)")
def yaml_validate(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {"error": "pyyaml not installed"}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {"valid": True, "type": type(data).__name__}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@tool("json_validate", "devops", "Validate a JSON file")
def json_validate(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return {"valid": True, "path": path}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@tool("env_file_check", "devops", "Check .env files exist and required keys present")
def env_file_check(path: str = ".env") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": path}
    keys = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0].strip())
    return {"exists": True, "path": path, "keys": keys, "count": len(keys)}


@tool("check_command_exists", "devops", "Check whether a CLI command is available on PATH")
def check_command_exists(command: str) -> Dict[str, Any]:
    return {"command": command, "exists": shutil.which(command) is not None}


@tool("which", "devops", "Show the path to an executable on PATH")
def which(command: str) -> Dict[str, Any]:
    return {"command": command, "path": shutil.which(command)}


@tool("where_python", "devops", "Show the location of the active Python interpreter")
def where_python() -> Dict[str, Any]:
    import sys

    return {"python": sys.executable}


@tool("check_ports_in_use", "devops", "List TCP ports currently in use (via ss/lsof)")
def check_ports_in_use() -> Dict[str, Any]:
    return _sh("ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || echo 'no ss/netstat'", timeout=20)


# ---------------------------------------------------------------------------
# Deploy / infra
# ---------------------------------------------------------------------------

@tool("systemctl_status", "devops", "Show systemd service status (if systemd is present)")
def systemctl_status(service: str) -> Dict[str, Any]:
    return _sh(f"systemctl status {service} --no-pager 2>/dev/null || echo 'systemd unavailable'", timeout=30)


@tool("systemctl_restart", "devops", "Restart a systemd service (requires privileges)")
def systemctl_restart(service: str) -> Dict[str, Any]:
    return _sh(f"systemctl restart {service} 2>/dev/null || echo 'systemd unavailable'", timeout=60)


@tool("crontab_list", "devops", "List the current user's crontab")
def crontab_list() -> Dict[str, Any]:
    return _sh("crontab -l 2>/dev/null || echo 'no crontab'")


@tool("tar_create", "devops", "Create a tar.gz archive of a directory")
def tar_create(source: str, archive: str) -> Dict[str, Any]:
    return _sh(f"tar -czf {archive} -C {Path(source).parent} {Path(source).name}", timeout=300)


@tool("zip_create", "devops", "Create a zip archive of a directory")
def zip_create(source: str, archive: str) -> Dict[str, Any]:
    return _sh(f"cd {Path(source).parent} && zip -r {archive} {Path(source).name}", timeout=300)


@tool("archive_extract", "devops", "Extract a tar.gz / zip archive")
def archive_extract(archive: str, dest: str = ".") -> Dict[str, Any]:
    if archive.endswith(".zip"):
        return _sh(f"unzip -o {archive} -d {dest}", timeout=300)
    return _sh(f"tar -xzf {archive} -C {dest}", timeout=300)


@tool("file_serve", "devops", "Check that a local HTTP server can serve a directory (python http.server)")
def file_serve(path: str = ".", port: int = 8000) -> Dict[str, Any]:
    return _sh(f"cd {path} && timeout 3 python -m http.server {port} --bind 127.0.0.1 2>&1 || true", timeout=10)


@tool("health_http", "devops", "HTTP GET a health endpoint and report status code + latency")
def health_http(url: str, timeout: int = 10) -> Dict[str, Any]:
    import time as _t
    import urllib.request

    start = _t.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            latency_ms = round((_t.time() - start) * 1000, 2)
            return {"url": url, "status": r.status, "latency_ms": latency_ms, "up": True}
    except Exception as e:
        latency_ms = round((_t.time() - start) * 1000, 2)
        return {"url": url, "error": str(e), "latency_ms": latency_ms, "up": False}


@tool("retry", "devops", "Retry a shell command N times until it succeeds")
def retry(command: str, times: int = 3, delay: int = 1) -> Dict[str, Any]:
    for i in range(1, times + 1):
        res = _sh(command, timeout=60)
        if res.get("returncode") == 0:
            res["attempts"] = i
            return res
        time.sleep(delay)
    res["attempts"] = times
    res["success"] = False
    return res


__all__ = []