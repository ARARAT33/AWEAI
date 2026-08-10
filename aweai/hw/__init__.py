from __future__ import annotations

import platform
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DeviceInfo:
    device_id: str
    device_type: str
    arch: str
    cores: int
    memory_gb: float
    gpus: List[Dict[str, Any]]
    accelerators: List[Dict[str, Any]]
    network: Dict[str, Any]
    storage: Dict[str, Any]
    tier: str
    capabilities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareAbstractionLayer:
    def __init__(self) -> None:
        self._devices: Dict[str, DeviceInfo] = {}
        self._current_device: Optional[DeviceInfo] = None

    def detect_current(self) -> DeviceInfo:
        import os
        import shutil
        arch = platform.machine()
        cores = os.cpu_count() or 1
        mem_gb = 0.0
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            mem_gb = 4.0
        gpus: List[Dict[str, Any]] = []
        accelerators: List[Dict[str, Any]] = []
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                import subprocess
                out = subprocess.run([nvidia_smi, "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append({"index": int(parts[0]), "name": parts[1], "memory_mb": float(parts[2]), "compute_cap": parts[3]})
            except Exception:
                pass
        if not gpus:
            gpus.append({"index": 0, "name": "cpu-fallback", "memory_mb": mem_gb * 1024, "compute_cap": "n/a"})
        tier = self._compute_tier(cores, mem_gb, len(gpus))
        capabilities = ["cpu", "storage", "network"]
        if gpus:
            capabilities.append("gpu")
        if any("TPU" in str(g.get("name", "")).upper() for g in gpus):
            capabilities.append("tpu")
        if any("NPU" in str(g.get("name", "")).upper() for g in gpus):
            capabilities.append("npu")
        hostname = socket.gethostname()
        device_id = f"{hostname}-{arch}-{cores}"
        self._current_device = DeviceInfo(
            device_id=device_id,
            device_type="host",
            arch=arch,
            cores=cores,
            memory_gb=mem_gb,
            gpus=gpus,
            accelerators=accelerators,
            network={"hostname": hostname, "interfaces": []},
            storage={"total_gb": mem_gb * 2},
            tier=tier,
            capabilities=capabilities,
        )
        self._devices[device_id] = self._current_device
        return self._current_device

    def _compute_tier(self, cores: int, mem_gb: float, gpu_count: int) -> str:
        if cores >= 256 and mem_gb >= 1024 and gpu_count >= 64:
            return "supercomputer"
        if cores >= 64 and mem_gb >= 256 and gpu_count >= 8:
            return "datacenter"
        if cores >= 16 and mem_gb >= 32 and gpu_count >= 1:
            return "workstation"
        if cores >= 4 and mem_gb >= 8:
            return "pc"
        return "edge"

    def get_current(self) -> Optional[DeviceInfo]:
        if self._current_device is None:
            self.detect_current()
        return self._current_device

    def register_remote(self, device: DeviceInfo) -> None:
        self._devices[device.device_id] = device

    def devices(self) -> List[DeviceInfo]:
        return list(self._devices.values())

    def summary(self) -> Dict[str, Any]:
        current = self.get_current()
        if not current:
            return {"tier": "unknown", "devices": 0}
        return {
            "tier": current.tier,
            "arch": current.arch,
            "cores": current.cores,
            "memory_gb": current.memory_gb,
            "gpus": len(current.gpus),
            "capabilities": current.capabilities,
            "devices": len(self._devices),
        }

    def to_dict(self) -> Dict[str, Any]:
        current = self.get_current()
        if current:
            return current.__dict__
        return {"error": "no device detected"}
