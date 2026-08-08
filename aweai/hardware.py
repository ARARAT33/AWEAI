"""Hardware detection and resource scoring.

Zero heavy dependencies: uses the standard library, and opportunistically
queries torch / nvidia-smi / psutil / platform when available.

The result is a dict consumed by the model selector to pick the best model
for the current machine (requirement #8: learn the resources and pick the
best model for them).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:  # pragma: no cover - optional
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False

try:  # pragma: no cover - optional
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        ).stdout
        return out.strip() or None
    except Exception:
        return None


@dataclass
class HardwareInfo:
    platform: str = "unknown"
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_vram_gb: List[float] = field(default_factory=list)
    gpu_total_vram_gb: float = 0.0
    torch_cuda: bool = False
    torch_mps: bool = False
    disk_free_gb: float = 0.0
    is_android: bool = False
    is_laptop: bool = False
    score: int = 0  # rough compute score: higher = more capable

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_free_gb": round(self.ram_free_gb, 1),
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "gpu_vram_gb": [round(v, 1) for v in self.gpu_vram_gb],
            "gpu_total_vram_gb": round(self.gpu_total_vram_gb, 1),
            "torch_cuda": self.torch_cuda,
            "torch_mps": self.torch_mps,
            "disk_free_gb": round(self.disk_free_gb, 1),
            "is_android": self.is_android,
            "is_laptop": self.is_laptop,
            "score": self.score,
            "recommended_tier": recommended_tier(self),
        }

    def summary(self) -> str:
        d = self.to_dict()
        gpu = ", ".join(
            f"{n} ({v}GB)" for n, v in zip(d["gpu_names"], d["gpu_vram_gb"])
        ) or "None"
        return (
            f"Platform: {d['platform']} | CPU: {d['cpu_count']} cores "
            f"({d['cpu_freq_mhz']:.0f} MHz) | RAM: {d['ram_total_gb']} GB "
            f"({d['ram_free_gb']} free) | GPU: {gpu} | "
            f"Score: {d['score']} (tier: {d['recommended_tier']})"
        )


def _cpu_count() -> int:
    if _HAS_PSUTIL:
        try:
            return psutil.cpu_count(logical=True) or os.cpu_count() or 0
        except Exception:
            pass
    return os.cpu_count() or 0


def _cpu_freq_mhz() -> float:
    if _HAS_PSUTIL:
        try:
            f = psutil.cpu_freq()
            if f:
                return float(getattr(f, "max", 0) or getattr(f, "current", 0) or 0)
        except Exception:
            pass
    return 0.0


def _ram() -> (float, float):
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
        except Exception:
            pass
    # Linux fallback
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            data = {}
            for line in f:
                k, v = line.split(":", 1)
                data[k] = int(v.strip().split()[0]) * 1024
        total = data.get("MemTotal", 0) / (1024 ** 3)
        avail = data.get("MemAvailable", data.get("MemFree", 0)) / (1024 ** 3)
        return total, avail
    except Exception:
        return 0.0, 0.0


def _disk_free_gb() -> float:
    try:
        st = os.statvfs("/")
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        return 0.0


def _gpu_info() -> (int, List[str], List[float], float):
    names: List[str] = []
    vram: List[float] = []
    total = 0.0

    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    names.append(name)
                    gb = float(props.total_memory) / (1024 ** 3)
                    vram.append(gb)
                    total += gb
            if torch.backends.mps.is_available():
                names.append("Apple MPS")
                vram.append(0.0)  # shared memory
        except Exception:
            pass

    if not names:
        nvidia = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if nvidia:
            for line in nvidia.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    try:
                        gb = float(parts[1]) / 1024.0
                    except ValueError:
                        gb = 0.0
                    names.append(parts[0])
                    vram.append(gb)
                    total += gb

    # Android GPU (Adreno/Mali) via /proc/gpu
    if not names and os.path.exists("/proc/gpu"):
        try:
            gpu = open("/proc/gpu", encoding="utf-8", errors="ignore").read().strip()
            if gpu:
                names.append(gpu.splitlines()[0])
                vram.append(0.0)
        except Exception:
            pass

    return len(names), names, vram, total


def recommended_tier(hw: "HardwareInfo") -> str:
    """Classify machine into a capability tier for the model selector."""
    if hw.gpu_total_vram_gb >= 24:
        return "large"
    if hw.gpu_total_vram_gb >= 8:
        return "medium-gpu"
    if hw.gpu_total_vram_gb >= 4:
        return "small-gpu"
    if hw.ram_total_gb >= 24 and hw.cpu_count >= 8:
        return "high-cpu"
    if hw.ram_total_gb >= 8:
        return "mid-cpu"
    return "low"


def detect() -> HardwareInfo:
    plat = platform.system().lower()
    if plat == "linux" and os.path.exists("/system/build.prop"):
        plat = "android"
    cpu = _cpu_count()
    freq = _cpu_freq_mhz()
    ram_total, ram_free = _ram()
    gpu_count, gpu_names, gpu_vram, gpu_total = _gpu_info()
    torch_cuda = bool(_HAS_TORCH and torch.cuda.is_available())
    torch_mps = bool(_HAS_TORCH and torch.backends.mps.is_available())

    hw = HardwareInfo(
        platform=plat,
        cpu_count=cpu,
        cpu_freq_mhz=freq,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        gpu_vram_gb=gpu_vram,
        gpu_total_vram_gb=gpu_total,
        torch_cuda=torch_cuda,
        torch_mps=torch_mps,
        disk_free_gb=_disk_free_gb(),
        is_android=(plat == "android"),
        is_laptop="BAT" in platform.system().upper() or "laptop" in platform.system().lower(),
    )

    # crude compute score
    score = cpu * (1 + freq / 4000.0)
    score += ram_total * 2
    score += gpu_total * 20
    if torch_cuda:
        score += 100
    if torch_mps:
        score += 60
    hw.score = int(score)
    return hw


def has_gpu() -> bool:
    return detect().gpu_count > 0
