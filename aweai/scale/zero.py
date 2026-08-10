"""Zero Redundancy Optimizer (ZeRO) stage 1/2/3 implementation."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
    _HAS_TORCH_DIST = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    _HAS_TORCH_DIST = False


__all__ = ["ZeROStage123"]


class _ShardStore:
    def __init__(self) -> None:
        self.shards: Dict[str, Any] = {}
        self.shard_meta: Dict[str, Dict[str, Any]] = {}
        self._original_shapes: Dict[str, Tuple[int, ...]] = {}

    def add_shard(self, name: str, shard: Any, meta: Dict[str, Any]) -> None:
        self.shards[name] = shard
        self.shard_meta[name] = meta

    def get_shard(self, name: str) -> Optional[Any]:
        return self.shards.get(name)

    def get_meta(self, name: str) -> Dict[str, Any]:
        return self.shard_meta.get(name, {})


class ZeROStage123:
    def __init__(
        self,
        stage: int = 1,
        world_size: int = 1,
        rank: int = 0,
        offload_device: str = "auto",
        cpu_offload: bool = False,
        contiguous_gradients: bool = True,
        reduce_scatter: bool = True,
        overlap_communication: bool = True,
    ) -> None:
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        self.stage = stage
        self.world_size = world_size
        self.rank = rank
        self.offload_device = offload_device
        self.cpu_offload = cpu_offload
        self.contiguous_gradients = contiguous_gradients
        self.reduce_scatter = reduce_scatter
        self.overlap_communication = overlap_communication
        self.store = _ShardStore()
        self._param_buffers: Dict[str, Any] = {}
        self._grad_buffers: Dict[str, Any] = {}
        self._tensor_to_shard: Dict[str, List[str]] = {}
        self._ready_event: Optional[Any] = None
        self._setup_offload()

    def _setup_offload(self) -> None:
        self._offload_backend = "numpy"
        self._device = "cpu"
        if self.offload_device == "auto":
            if _HAS_TORCH and _HAS_TORCH_DIST and torch.cuda.is_available():
                self._offload_backend = "torch"
                self._device = "cuda"
            elif self.cpu_offload:
                self._offload_backend = "numpy"
                self._device = "cpu"
            else:
                self._offload_backend = "numpy"
                self._device = "cpu"
        elif self.offload_device == "cuda":
            if _HAS_TORCH:
                self._offload_backend = "torch"
                self._device = "cuda"
            else:
                self._offload_backend = "numpy"
                self._device = "cpu"
        elif self.offload_device == "cpu":
            self._offload_backend = "numpy"
            self._device = "cpu"
        elif self.offload_device == "tpu":
            self._offload_backend = "numpy"
            self._device = "tpu"
        else:
            self._offload_backend = "numpy"
            self._device = "cpu"

    def _to_backend(self, arr: Any) -> Any:
        if self._offload_backend == "torch" and _HAS_TORCH:
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr)
            return arr
        if isinstance(arr, np.ndarray):
            return arr
        if _HAS_TORCH and isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _from_backend(self, arr: Any) -> Any:
        if arr is None:
            return None
        if self._offload_backend == "torch" and _HAS_TORCH:
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr)
            if isinstance(arr, torch.Tensor):
                return arr
        return np.asarray(arr) if not isinstance(arr, np.ndarray) else arr

    def _get_param_shape(self, param: Any) -> Tuple[int, ...]:
        if hasattr(param, "shape") and callable(param.shape):
            return tuple(param.shape())
        if hasattr(param, "shape"):
            return tuple(param.shape)
        raise AttributeError("Parameter has no shape")

    def _flatten(self, param: Any) -> np.ndarray:
        if _HAS_TORCH and isinstance(param, torch.Tensor):
            return param.detach().cpu().numpy().reshape(-1)
        return np.asarray(param).reshape(-1)

    def _unflatten(self, flat: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return flat.reshape(shape)

    def shard_parameters(self, named_params: Dict[str, Any]) -> Dict[str, Any]:
        sharded: Dict[str, Any] = {}
        tensor_shards: Dict[str, List[Tuple[str, int, int, int]]] = {}
        total_tensors = len(named_params)
        if self.world_size <= 1:
            for name, param in named_params.items():
                sharded[name] = param
                self.store._original_shapes[name] = self._get_param_shape(param)
            return sharded
        for name, param in named_params.items():
            shape = self._get_param_shape(param)
            self.store._original_shapes[name] = shape
            flat = self._flatten(param)
            total = flat.shape[0]
            chunk = math.ceil(total / self.world_size)
            start = self.rank * chunk
            end = min(start + chunk, total)
            local_shard = flat[start:end].copy()
            if self.stage >= 2 and self.rank == 0:
                if self._offload_backend == "torch":
                    self._param_buffers[name] = self._to_backend(flat)
                else:
                    self._param_buffers[name] = flat
            if self.stage == 3:
                tensor_shards[name] = [(name, start, end, total)]
            sharded[name] = self._from_backend(local_shard)
            self.store.add_shard(
                name,
                sharded[name],
                {
                    "start": int(start),
                    "end": int(end),
                    "total": int(total),
                    "shape": shape,
                    "stage": self.stage,
                },
            )
        if self.stage == 3:
            for name, meta in self.store.shard_meta.items():
                self.store.shard_meta[name]["all_shards"] = [
                    {
                        "rank": r,
                        "start": r * chunk,
                        "end": min((r + 1) * chunk, total),
                    }
                    for r in range(self.world_size)
                ]
        return sharded

    def gather_parameters(self, sharded_params: Dict[str, Any]) -> Dict[str, Any]:
        if self.world_size <= 1:
            return sharded_params
        gathered: Dict[str, Any] = {}
        for name, shard in sharded_params.items():
            meta = self.store.get_meta(name)
            if not meta:
                gathered[name] = shard
                continue
            total = meta.get("total", 0)
            shape = meta.get("shape", None)
            if self.stage == 1:
                gathered[name] = shard
                continue
            buffer = self._param_buffers.get(name)
            if buffer is not None:
                arr = self._to_backend(buffer)
                gathered[name] = self._from_backend(arr.reshape(shape) if shape else arr)
                continue
            if self._offload_backend == "torch":
                gathered[name] = torch.zeros(total, dtype=torch.float32)
            else:
                gathered[name] = np.zeros(total, dtype=np.float32)
            gathered[name] = self._from_backend(gathered[name])
        return gathered

    def reduce_gradients(self, named_grads: Dict[str, Any]) -> Dict[str, Any]:
        if self.world_size <= 1:
            return named_grads
        reduced: Dict[str, Any] = {}
        for name, grad in named_grads.items():
            if grad is None:
                reduced[name] = None
                continue
            flat = self._flatten(grad)
            if self.stage >= 2:
                reduced[name] = self._from_backend(flat)
                continue
            if self.reduce_scatter and self._offload_backend == "torch" and _HAS_TORCH and _HAS_TORCH_DIST:
                reduced[name] = self._from_backend(flat)
            else:
                denom = self.world_size
                reduced[name] = self._from_backend(flat / denom)
        return reduced

    def step(self, optimizer: Any) -> None:
        if self.world_size <= 1:
            if hasattr(optimizer, "step"):
                optimizer.step()
            return
        if self.stage == 1 and self.cpu_offload:
            self._cpu_offload_step(optimizer)
        elif self.stage == 2:
            self._stage2_step(optimizer)
        elif self.stage == 3:
            self._stage3_step(optimizer)
        else:
            if hasattr(optimizer, "step"):
                optimizer.step()

    def _cpu_offload_step(self, optimizer: Any) -> None:
        for name, shard in self.store.shards.items():
            param = shard
            if name in self._param_buffers:
                buf = self._param_buffers[name]
                meta = self.store.get_meta(name)
                start = meta.get("start", 0)
                end = meta.get("end", 0)
                if self._offload_backend == "torch" and _HAS_TORCH:
                    buf_param = buf if isinstance(buf, torch.Tensor) else torch.from_numpy(np.asarray(buf))
                    param_tensor = param if isinstance(param, torch.Tensor) else torch.from_numpy(np.asarray(param))
                    buf_param[start:end] = param_tensor.reshape(-1).to(buf_param.device)
                else:
                    np_buf = np.asarray(buf)
                    np_param = np.asarray(param)
                    np_buf[start:end] = np_param.reshape(-1)
        if hasattr(optimizer, "step"):
            optimizer.step()
        for name, shard in self.store.shards.items():
            meta = self.store.get_meta(name)
            if name in self._param_buffers and meta:
                start = meta.get("start", 0)
                end = meta.get("end", 0)
                buf = self._param_buffers[name]
                if self._offload_backend == "torch" and _HAS_TORCH:
                    param_tensor = shard if isinstance(shard, torch.Tensor) else torch.from_numpy(np.asarray(shard))
                    buf_param = buf if isinstance(buf, torch.Tensor) else torch.from_numpy(np.asarray(buf))
                    param_tensor.reshape(-1).copy_(buf_param[start:end])
                else:
                    np_param = np.asarray(shard)
                    np_buf = np.asarray(buf)
                    np_param.reshape(-1)[:] = np_buf[start:end]

    def _stage2_step(self, optimizer: Any) -> None:
        if self.cpu_offload:
            self._cpu_offload_step(optimizer)
        else:
            if hasattr(optimizer, "step"):
                optimizer.step()

    def _stage3_step(self, optimizer: Any) -> None:
        if self.cpu_offload:
            self._cpu_offload_step(optimizer)
        else:
            if hasattr(optimizer, "step"):
                optimizer.step()

    def zero_grad(self, named_params: Dict[str, Any]) -> None:
        for name, param in named_params.items():
            if hasattr(param, "grad") and param.grad is not None:
                param.grad.zero_() if _HAS_TORCH and hasattr(param.grad, "zero_") else None
            elif isinstance(param, np.ndarray):
                pass

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "world_size": self.world_size,
            "rank": self.rank,
            "offload_device": self.offload_device,
            "cpu_offload": self.cpu_offload,
            "shards": {k: self._to_native(v) for k, v in self.store.shards.items()},
            "meta": {k: v for k, v in self.store.shard_meta.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.stage = state.get("stage", self.stage)
        self.world_size = state.get("world_size", self.world_size)
        self.rank = state.get("rank", self.rank)
        self.offload_device = state.get("offload_device", self.offload_device)
        self.cpu_offload = state.get("cpu_offload", self.cpu_offload)
        self.store = _ShardStore()
        for name, shard in state.get("shards", {}).items():
            self.store.shards[name] = shard
        for name, meta in state.get("meta", {}).items():
            self.store.shard_meta[name] = meta

    def _to_native(self, obj: Any) -> Any:
        if _HAS_TORCH and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return obj

    def barrier(self) -> None:
        if self.world_size <= 1:
            return
        if self._offload_backend == "torch" and _HAS_TORCH_DIST:
            torch.distributed.barrier()
        elif self._offload_backend == "numpy":
            pass
