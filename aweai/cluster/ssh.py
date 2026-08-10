"""SSH remote management with multi-hop support and tunnels."""

from __future__ import annotations

import os
import select
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SSHHost:
    host: str
    port: int = 22
    user: str = "root"
    key_path: Optional[str] = None
    password: Optional[str] = None
    timeout: float = 30.0
    proxy_jump: Optional["SSHHost"] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def connection_string(self) -> str:
        parts = [f"{self.user}@{self.host}"]
        if self.port != 22:
            parts.append(f"-p {self.port}")
        if self.key_path:
            parts.append(f"-i {self.key_path}")
        if self.proxy_jump:
            parts.append(f"-J {self.proxy_jump.user}@{self.proxy_jump.host}:{self.proxy_jump.port}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "key_path": self.key_path,
            "timeout": self.timeout,
            "proxy_jump": self.proxy_jump.connection_string() if self.proxy_jump else None,
            "labels": self.labels,
        }


@dataclass
class SSHTunnel:
    local_port: int
    remote_host: str
    remote_port: int
    via: SSHHost
    bind_address: str = "127.0.0.1"
    active: bool = False
    process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        cmd = [
            "ssh", "-N", "-L",
            f"{self.bind_address}:{self.local_port}:{self.remote_host}:{self.remote_port}",
            f"{self.via.user}@{self.via.host}", "-p", str(self.via.port),
        ]
        if self.via.key_path:
            cmd.extend(["-i", self.via.key_path])
        if self.via.proxy_jump:
            cmd.extend(["-o", f"ProxyJump={self.via.proxy_jump.user}@{self.via.proxy_jump.host}:{self.via.proxy_jump.port}"])
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.5)
            self.active = True
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        self.active = False

    def is_alive(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        self.active = False
        return False


class SSHManager:
    def __init__(self) -> None:
        self._hosts: Dict[str, SSHHost] = {}
        self._tunnels: Dict[str, SSHTunnel] = {}
        self._lock = threading.Lock()

    def register_host(self, name: str, host: SSHHost) -> None:
        self._hosts[name] = host

    def get_host(self, name: str) -> Optional[SSHHost]:
        return self._hosts.get(name)

    def execute(self, host: SSHHost, command: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        timeout = timeout if timeout is not None else host.timeout
        cmd = ["ssh"]
        cmd.extend(["-o", "StrictHostKeyChecking=no"])
        cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])
        if host.key_path:
            cmd.extend(["-i", host.key_path])
        if host.proxy_jump:
            cmd.extend(["-o", f"ProxyJump={host.proxy_jump.user}@{host.proxy_jump.host}:{host.proxy_jump.port}"])
        cmd.extend([f"{host.user}@{host.host}", "-p", str(host.port), command])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
                "host": host.host,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": "timeout", "command": command, "host": host.host}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "command": command, "host": host.host}

    def execute_multi(self, hosts: List[SSHHost], command: str, timeout: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        threads = []
        def _run(h: SSHHost) -> None:
            results[h.host] = self.execute(h, command, timeout=timeout)
        for h in hosts:
            t = threading.Thread(target=_run, args=(h,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return results

    def create_tunnel(self, name: str, tunnel: SSHTunnel) -> bool:
        with self._lock:
            if name in self._tunnels:
                self._tunnels[name].stop()
            success = tunnel.start()
            if success:
                self._tunnels[name] = tunnel
            return success

    def close_tunnel(self, name: str) -> None:
        with self._lock:
            tunnel = self._tunnels.pop(name, None)
            if tunnel:
                tunnel.stop()

    def sync_directory(self, src_host: SSHHost, src_path: str, dst_host: SSHHost, dst_path: str, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        exclude = exclude or []
        cmd = ["rsync", "-avz", "--delete", "-e", f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {src_host.port}"]
        for pat in exclude:
            cmd.extend(["--exclude", pat])
        cmd.append(f"{src_host.user}@{src_host.host}:{src_path}")
        cmd.append(f"{dst_host.user}@{dst_host.host}:{dst_path}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    def remote_put(self, host: SSHHost, local_path: str, remote_path: str) -> Dict[str, Any]:
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-P", str(host.port)]
        if host.key_path:
            cmd.extend(["-i", host.key_path])
        cmd.extend([local_path, f"{host.user}@{host.host}:{remote_path}"])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    def remote_get(self, host: SSHHost, remote_path: str, local_path: str) -> Dict[str, Any]:
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-P", str(host.port)]
        if host.key_path:
            cmd.extend(["-i", host.key_path])
        cmd.extend([f"{host.user}@{host.host}:{remote_path}", local_path])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    def shell(self, host: SSHHost, interactive: bool = False) -> int:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
        if host.key_path:
            cmd.extend(["-i", host.key_path])
        if host.proxy_jump:
            cmd.extend(["-o", f"ProxyJump={host.proxy_jump.user}@{host.proxy_jump.host}:{host.proxy_jump.port}"])
        cmd.extend([f"{host.user}@{host.host}", "-p", str(host.port)])
        if interactive:
            return os.system(" ".join(cmd))
        return 0

    def list_tunnels(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "local_port": t.local_port,
                "remote_host": t.remote_host,
                "remote_port": t.remote_port,
                "active": t.is_alive(),
            }
            for name, t in self._tunnels.items()
        ]

    def close_all_tunnels(self) -> None:
        for tunnel in list(self._tunnels.values()):
            tunnel.stop()
        self._tunnels.clear()
