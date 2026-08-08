"""Hardware detection and resource scoring.

Zero heavy dependencies: uses the standard library, and opportunistically
queries torch / nvidia-smi / psutil / platform when available.

The result is consumed by the resource-adaptive engine (aweai/selector.py)
which picks the best model type and size for the current machine.
"""

from __future__ import annotations

import os
import platform
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

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_free_gb": round(self.ram_free_gb, 2),
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "gpu_vram_gb": self.gpu_vram_gb,
            "gpu_total_vram_gb": round(self.gpu_total_vram_gb, 2),
            "torch_cuda": self.torch_cuda,
            "torch_mps": self.torch_mps,
            "disk_free_gb": round(self.disk_free_gb, 2),
            "is_android": self.is_android,
            "is_laptop": self.is_laptop,
            "tier": tier_of(self),
        }


def _free_ram_gb() -> float:
    try:
        if _HAS_PSUTIL:
            return round(psutil.virtual_memory().available / (1024 ** 3), 2)
        if os.name == "posix":
            out = _run(["free", "-b"])
            if out:
                for line in out.splitlines():
                    if line.lower().startswith("mem:"):
                        parts = line.split()
                        if len(parts) >= 7:
                            return round(int(parts[6]) / (1024 ** 3), 2)
        return 0.0
    except Exception:
        return 0.0


def _total_ram_gb() -> float:
    try:
        if _HAS_PSUTIL:
            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        if os.name == "posix":
            out = _run(["free", "-b"])
            if out:
                for line in out.splitlines():
                    if line.lower().startswith("mem:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return round(int(parts[1]) / (1024 ** 3), 2)
        return 0.0
    except Exception:
        return 0.0


def _cpu_count() -> int:
    try:
        if _HAS_PSUTIL:
            return psutil.cpu_count(logical=True) or 0
        return os.cpu_count() or 0
    except Exception:
        return 0


def _cpu_freq_mhz() -> float:
    try:
        if _HAS_PSUTIL:
            return float(psutil.cpu_freq().current or 0.0)
        return 0.0
    except Exception:
        return 0.0


def _disk_free_gb() -> float:
    try:
        if _HAS_PSUTIL:
            return round(psutil.disk_usage("/").free / (1024 ** 3), 2)
        return 0.0
    except Exception:
        return 0.0


def detect() -> HardwareInfo:
    """Detect hardware and return a HardwareInfo dataclass."""
    info = HardwareInfo()
    info.platform = platform.system().lower()
    info.cpu_count = _cpu_count()
    info.cpu_freq_mhz = _cpu_freq_mhz()
    info.ram_total_gb = _total_ram_gb()
    info.ram_free_gb = _free_ram_gb()
    info.disk_free_gb = _disk_free_gb()

    if info.platform == "linux":
        info.is_android = os.environ.get("ANDROID_ARGUMENT") is not None or os.path.exists("/system/build.prop")
    else:
        info.is_android = False

    if _HAS_TORCH:
        try:
            info.torch_cuda = bool(torch.cuda.is_available())
            info.torch_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            pass

    if info.torch_cuda:
        try:
            info.gpu_count = torch.cuda.device_count()
            for i in range(info.gpu_count):
                name = torch.cuda.get_device_name(i)
                info.gpu_names.append(str(name))
                try:
                    vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    info.gpu_vram_gb.append(round(float(vram), 2))
                except Exception:
                    info.gpu_vram_gb.append(0.0)
            info.gpu_total_vram_gb = round(sum(info.gpu_vram_gb), 2)
        except Exception:
            pass
    elif info.platform == "linux":
        nvidia = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if nvidia:
            for line in nvidia.splitlines():
                try:
                    name, mem = [p.strip() for p in line.split(",")]
                    info.gpu_names.append(name)
                    info.gpu_vram_gb.append(round(float(mem) / 1024.0, 2))
                except Exception:
                    pass
            info.gpu_count = len(info.gpu_names)
            info.gpu_total_vram_gb = round(sum(info.gpu_vram_gb), 2)

    if info.gpu_count == 0 and info.cpu_count <= 8 and info.ram_total_gb <= 16:
        info.is_laptop = True
    return info


def tier_of(info: HardwareInfo) -> str:
    """Return a coarse resource tier: 'edge' | 'laptop' | 'desktop' | 'gpu'."""
    if info.gpu_count > 0 and info.gpu_total_vram_gb >= 6:
        return "gpu"
    if info.ram_total_gb >= 32 and info.cpu_count >= 8:
        return "desktop"
    if info.ram_total_gb >= 8:
        return "laptop"
    return "edge"


def best_device() -> str:
    """Pick the best torch device for this machine."""
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"
