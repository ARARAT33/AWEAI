"""CPU/SSD/NVMe Offloading engine for models larger than RAM."""

from __future__ import annotations

import asyncio
import os
import pickle
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


__all__ = ["OffloadEngine"]


class _MemoryBlock:
    def __init__(self, key: str, data: Any, device: str = "cpu", size_bytes: int = 0) -> None:
        self.key = key
        self.data = data
        self.device = device
        self.size_bytes = size_bytes
        self.last_accessed: float = time.time()
        self.pinned: bool = False
        self.dirty: bool = False

    def touch(self) -> None:
        self.last_accessed = time.time()


class _SwapFile:
    def __init__(self, path: str, max_size_bytes: int) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self._files: Dict[str, str] = {}
        self._lock = threading.Lock()

    def write(self, key: str, data: Any) -> None:
        fpath = self.path / f"{key}.swap"
        with self._lock:
            self._files[key] = str(fpath)
            tmp = str(fpath) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, str(fpath))

    def read(self, key: str) -> Any:
        fpath = Path(self._files.get(key, ""))
        if not fpath.exists():
            return None
        with open(fpath, "rb") as f:
            return pickle.load(f)

    def delete(self, key: str) -> None:
        with self._lock:
            fpath = self._files.pop(key, None)
            if fpath and Path(fpath).exists():
                try:
                    Path(fpath).unlink()
                except Exception:
                    pass

    def exists(self, key: str) -> bool:
        return key in self._files and Path(self._files[key]).exists()

    def current_size_bytes(self) -> int:
        total = 0
        with self._lock:
            for fpath in self._files.values():
                try:
                    total += Path(fpath).stat().st_size
                except Exception:
                    pass
        return total

    def shutdown(self) -> None:
        with self._lock:
            for key in list(self._files.keys()):
                self.delete(key)


class OffloadEngine:
    def __init__(
        self,
        ram_limit_fraction: float = 0.8,
        swap_dir: Optional[str] = None,
        ssd_enabled: bool = True,
        nvme_enabled: bool = True,
        prefetch_ratio: float = 0.1,
        eviction_policy: str = "lru",
        async_prefetch: bool = True,
        prefetch_threads: int = 2,
        max_prefetch_queue: int = 16,
    ) -> None:
        self.ram_limit_fraction = ram_limit_fraction
        self._swap_dir = swap_dir or tempfile.mkdtemp(prefix="aweai_offload_")
        self.ssd_enabled = ssd_enabled
        self.nvme_enabled = nvme_enabled
        self.prefetch_ratio = prefetch_ratio
        self.eviction_policy = eviction_policy
        self.async_prefetch = async_prefetch
        self.prefetch_threads = prefetch_threads
        self.max_prefetch_queue = max_prefetch_queue
        self._hot_store: OrderedDict[str, _MemoryBlock] = OrderedDict()
        self._swap = _SwapFile(self._swap_dir, max_size_bytes=self._max_swap_bytes())
        self._lock = threading.RLock()
        self._prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=max_prefetch_queue)
        self._prefetch_loop_running = False
        self._prefetch_event: Optional[asyncio.Event] = None
        self._total_allocated: int = 0
        self._ram_capacity: int = self._detect_ram_capacity()
        self._nvme_capacity: int = self._detect_nvme_capacity()
        self._hierarchy: List[str] = self._build_hierarchy()
        self._tensor_counter: int = 0

    def _detect_ram_capacity(self) -> int:
        try:
            import psutil  # type: ignore
            return int(psutil.virtual_memory().total * self.ram_limit_fraction)
        except Exception:
            return int(4 * 1024 ** 3)

    def _detect_nvme_capacity(self) -> int:
        try:
            st = os.statvfs("/")
            return int(st.f_bavail * st.f_frsize * self.ram_limit_fraction)
        except Exception:
            return int(256 * 1024 ** 3)

    def _max_swap_bytes(self) -> int:
        try:
            st = os.statvfs(self._swap_dir)
            return int(st.f_bavail * st.f_frsize * self.ram_limit_fraction)
        except Exception:
            return int(128 * 1024 ** 3)

    def _build_hierarchy(self) -> List[str]:
        hierarchy = ["ram"]
        if self.nvme_enabled and self._nvme_capacity > 0:
            hierarchy.append("nvme")
        elif self.ssd_enabled:
            hierarchy.append("ssd")
        hierarchy.append("swap")
        return hierarchy

    def register(self, name: str, data: Any, device: str = "cpu") -> None:
        with self._lock:
            if name in self._hot_store:
                self._hot_store.move_to_end(name)
                self._hot_store[name].data = data
                self._hot_store[name].touch()
                return
            size = self._estimate_size(data)
            if self._total_allocated + size > self._ram_capacity:
                evicted = self._evict_until(size)
                if not evicted:
                    self._swap.write(name, data)
                    return
            block = _MemoryBlock(name, data, device=device, size_bytes=size)
            self._hot_store[name] = block
            self._total_allocated += size

    def get(self, name: str) -> Optional[Any]:
        with self._lock:
            block = self._hot_store.get(name)
            if block is not None:
                block.touch()
                self._hot_store.move_to_end(name)
                return block.data
            if self._swap.exists(name):
                data = self._swap.read(name)
                if data is not None:
                    size = self._estimate_size(data)
                    if self._total_allocated + size <= self._ram_capacity:
                        block = _MemoryBlock(name, data, device="cpu", size_bytes=size)
                        self._hot_store[name] = block
                        self._total_allocated += size
                    return data
            return None

    def prefetch(self, name: str) -> None:
        if not self.async_prefetch:
            self.get(name)
            return
        try:
            self._prefetch_queue.put_nowait(name)
        except asyncio.QueueFull:
            pass

    async def _prefetch_worker(self) -> None:
        while self._prefetch_loop_running:
            try:
                name = await asyncio.wait_for(self._prefetch_queue.get(), timeout=0.1)
                self.get(name)
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue

    async def start_prefetch_loop(self) -> None:
        self._prefetch_loop_running = True
        self._prefetch_event = asyncio.Event()
        if self.async_prefetch:
            asyncio.create_task(self._prefetch_worker())

    async def stop_prefetch_loop(self) -> None:
        self._prefetch_loop_running = False
        if self._prefetch_event:
            self._prefetch_event.set()

    def unregister(self, name: str) -> None:
        with self._lock:
            block = self._hot_store.pop(name, None)
            if block is not None:
                self._total_allocated -= block.size_bytes
            self._swap.delete(name)

    def _evict_until(self, needed: int) -> bool:
        if self.eviction_policy == "lru":
            return self._evict_lru(needed)
        return self._evict_lfu(needed)

    def _evict_lru(self, needed: int) -> bool:
        freed = 0
        while self._total_allocated + needed > self._ram_capacity and self._hot_store:
            name, block = self._hot_store.popitem(last=False)
            freed += block.size_bytes
            self._total_allocated -= block.size_bytes
            if not block.pinned:
                self._swap.write(name, block.data)
        return freed >= needed

    def _evict_lfu(self, needed: int) -> bool:
        items = list(self._hot_store.items())
        items.sort(key=lambda item: item[1].last_accessed)
        freed = 0
        for name, block in items:
            if self._total_allocated + needed <= self._ram_capacity:
                break
            if not block.pinned:
                del self._hot_store[name]
                freed += block.size_bytes
                self._total_allocated -= block.size_bytes
                self._swap.write(name, block.data)
        return freed >= needed

    def _estimate_size(self, data: Any) -> int:
        if data is None:
            return 0
        if _HAS_TORCH and isinstance(data, torch.Tensor):
            return int(data.numel() * data.element_size())
        if isinstance(data, np.ndarray):
            return int(data.size * data.dtype.itemsize)
        if isinstance(data, (list, tuple)):
            return sum(self._estimate_size(x) for x in data)
        if isinstance(data, dict):
            return sum(self._estimate_size(v) for v in data.values())
        return 1024

    def to(self, name: str, device: str) -> None:
        with self._lock:
            block = self._hot_store.get(name)
            if block is None:
                return
            block.device = device
            block.touch()

    def pin(self, name: str) -> None:
        with self._lock:
            if name in self._hot_store:
                self._hot_store[name].pinned = True
                self._hot_store[name].touch()

    def unpin(self, name: str) -> None:
        with self._lock:
            if name in self._hot_store:
                self._hot_store[name].pinned = False

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hot_count": len(self._hot_store),
                "swap_count": sum(1 for _ in self._swap._files),
                "total_allocated_bytes": self._total_allocated,
                "ram_capacity_bytes": self._ram_capacity,
                "nvme_capacity_bytes": self._nvme_capacity,
                "swap_usage_bytes": self._swap.current_size_bytes(),
                "hierarchy": self._hierarchy,
            }

    def clear(self) -> None:
        with self._lock:
            self._hot_store.clear()
            self._total_allocated = 0
        self._swap.shutdown()
        self._swap = _SwapFile(self._swap_dir, max_size_bytes=self._max_swap_bytes())

    def checkpoint(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with self._lock:
            meta: Dict[str, Any] = {
                "hot_keys": list(self._hot_store.keys()),
                "swap_files": dict(self._swap._files),
                "hierarchy": self._hierarchy,
            }
            for name, block in self._hot_store.items():
                fpath = Path(path) / f"{name}.pkl"
                with open(fpath, "wb") as f:
                    pickle.dump(block.data, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(Path(path) / "offload_meta.pkl", "wb") as f:
                pickle.dump(meta, f)

    def restore(self, path: str) -> None:
        base = Path(path)
        if not base.exists():
            return
        meta_path = base / "offload_meta.pkl"
        if not meta_path.exists():
            return
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._hot_store.clear()
        self._total_allocated = 0
        for name in meta.get("hot_keys", []):
            fpath = base / f"{name}.pkl"
            if fpath.exists():
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                size = self._estimate_size(data)
                self._hot_store[name] = _MemoryBlock(name, data, device="cpu", size_bytes=size)
                self._total_allocated += size
        for name, fpath in meta.get("swap_files", {}).items():
            if Path(fpath).exists():
                self._swap._files[name] = fpath

    def shutdown(self) -> None:
        self.clear()
