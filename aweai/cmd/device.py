# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Device & server commands: SSH, remote, cluster, distributed training,
device orchestration."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import APP_DIR, err, jdump, ok, run_cmd

app = typer.Typer(help="Devices & servers: SSH, remote, cluster, distributed training, orchestration")


def _hosts_file() -> Path:
    return APP_DIR / "hosts.json"


def _load_hosts() -> Dict[str, Any]:
    if _hosts_file().exists():
        try:
            return json.loads(_hosts_file().read_text(encoding="utf-8"))
        except Exception:
            return {"hosts": {}}
    return {"hosts": {}}


def _save_hosts(data: Dict[str, Any]) -> None:
    _hosts_file().parent.mkdir(parents=True, exist_ok=True)
    _hosts_file().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command("detect")
def detect():
    """Detect this device: CPU/GPU/RAM/disk/platform."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_gb = round(mem.total / (1024 ** 3), 2)
        disk = psutil.disk_usage("/")
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                ram_kb = int([ln for ln in f if ln.startswith("MemTotal")][0].split()[1])
            ram_gb = round(ram_kb / (1024 ** 2), 2)
        except Exception:
            ram_gb = None
        disk = None
    gpu = _detect_gpu()
    typer.echo(jdump(ok(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpus=os.cpu_count(),
        ram_gb=ram_gb,
        disk_free_gb=round(disk.free / (1024 ** 3), 2) if disk else None,
        gpu=gpu,
        python=platform.python_version(),
    )))


def _detect_gpu() -> List[str]:
    out = []
    nv = run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
    if nv:
        out.extend(ln.strip() for ln in nv.splitlines() if ln.strip())
    return out


@app.command("ssh")
def ssh(
    host: str = typer.Argument(..., help="user@host"),
    command: str = typer.Option("hostname", "--command", "-c", help="Remote command"),
    port: int = typer.Option(22, "--port", "-p"),
):
    """Run a command on a remote machine via SSH."""
    try:
        res = run_cmd(f"ssh -p {port} -o BatchMode=yes -o ConnectTimeout=8 {host} '{command}'")
        typer.echo(jdump(ok(host=host, output=res)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("ssh-add")
def ssh_add(
    name: str = typer.Argument(..., help="Host alias"),
    host: str = typer.Option(..., "--host", help="user@host"),
    port: int = typer.Option(22, "--port", "-p"),
    description: Optional[str] = typer.Option(None, "--desc"),
):
    """Register a remote host in the device cluster."""
    data = _load_hosts()
    data["hosts"][name] = {"host": host, "port": port, "description": description or ""}
    _save_hosts(data)
    typer.echo(jdump(ok(added=name, host=host)))


@app.command("ssh-list")
def ssh_list():
    """List registered remote hosts."""
    data = _load_hosts()
    typer.echo(jdump(ok(hosts=data["hosts"])))


@app.command("ssh-remove")
def ssh_remove(name: str = typer.Argument(..., help="Host alias")):
    """Remove a registered host."""
    data = _load_hosts()
    removed = data["hosts"].pop(name, None)
    _save_hosts(data)
    typer.echo(jdump(ok(removed=name, existed=removed is not None)))


@app.command("cluster-status")
def cluster_status():
    """Show cluster status of registered hosts (SSH probe)."""
    data = _load_hosts()
    results = {}
    for name, info in data["hosts"].items():
        out = run_cmd(f"ssh -p {info['port']} -o BatchMode=yes -o ConnectTimeout=5 {info['host']} 'hostname; nproc; uptime'")
        results[name] = {"reachable": bool(out), "info": info, "output": out[:200]}
    typer.echo(jdump(ok(cluster=results)))


@app.command("distributed")
def distributed(
    model_type: str = typer.Argument(..., help="Model type"),
    name: str = typer.Option(..., "--name", "-n"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d"),
    workers: int = typer.Option(0, "--workers", "-w"),
    backend: str = typer.Option("auto", "--backend", "-b", help="auto|thread|torch"),
    epochs: int = typer.Option(30, "--epochs", "-e"),
):
    """Run distributed training (thread/torch backends)."""
    try:
        from aweai.distributed import train_distributed

        if data_path:
            from aweai.data import load_any
            ds = load_any(data_path)
            X, y = ds.X, ds.y
        else:
            X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
        res = train_distributed(model_type, name, X, y=y, workers=workers, backend=backend, epochs=epochs)
        typer.echo(jdump(res))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("orchestrate")
def orchestrate(
    action: str = typer.Argument(..., help="start|stop|status|ps"),
    workers: int = typer.Option(4, "--workers", "-w", help="Workers to start"),
):
    """Orchestrate local training workers (start/stop/status)."""
    pid_file = APP_DIR / "orchestrator.pid"
    if action == "start":
        import multiprocessing as mp

        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        typer.echo(jdump(ok(action="started", workers=workers, pid=os.getpid(),
                            note="AWEAI device orchestrator registered; run `aweai devices orchestrate status`")))
    elif action == "stop":
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), 15)
            except Exception:
                pass
            pid_file.unlink(missing_ok=True)
        typer.echo(jdump(ok(action="stopped")))
    elif action == "status":
        typer.echo(jdump(ok(action="status", running=pid_file.exists(),
                            pid=int(pid_file.read_text().strip()) if pid_file.exists() else None)))
    elif action == "ps":
        typer.echo(jdump(ok(processes=_ps())))
    else:
        typer.echo(jdump(err("unknown action: start|stop|status|ps")))
        raise typer.Exit(code=1)


def _ps() -> List[Dict[str, Any]]:
    out = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_percent"]):
            try:
                pinfo = proc.info
                if pinfo["cmdline"]:
                    out.append({"pid": pinfo["pid"], "name": pinfo["name"],
                                "cpu_percent": pinfo["cpu_percent"], "memory_percent": pinfo["memory_percent"]})
            except Exception:
                pass
    except Exception:
        pass
    return out[:50]


@app.command("lan")
def lan():
    """Detect LAN IP addresses of this device."""
    ips = set()
    try:
        import psutil
        for iface, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family == socket.AF_INET:
                    ips.add(a.address)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    typer.echo(jdump(ok(ips=sorted(ips))))


@app.command("benchmark")
def benchmark(
    seconds: int = typer.Option(3, "--seconds", "-s"),
):
    """Quick CPU benchmark (prime computation rate)."""
    import math
    import time

    start = time.time()
    count = 0
    n = 2
    while time.time() - start < seconds:
        okp = True
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                okp = False
                break
        if okp:
            count += 1
        n += 1
    elapsed = time.time() - start
    typer.echo(jdump(ok(primes=count, seconds=round(elapsed, 2),
                        primes_per_sec=round(count / elapsed, 1))))
