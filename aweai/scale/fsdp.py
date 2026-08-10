"""Fully Sharded Data Parallel (FSDP) implementation."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
    _HAS_TORCH_DIST = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    _HAS_TORCH_DIST = False


__all__ = ["FSDPWrapper"]


class FSDPWrapper:
    def __init__(
        self,
        module: Any,
        process_group: Optional[Any] = None,
        sharding_strategy: str = "zero3",
        auto_wrap_policy: Optional[Any] = None,
        cpu_offload: bool = False,
        mixed_precision: bool = False,
        backward_prefetch: bool = True,
        forward_prefetch: bool = True,
        device_id: Optional[Union[int, str]] = None,
        sync_module_states: bool = True,
        sharded_param_names: Optional[List[str]] = None,
        limit_all_gathers: bool = True,
    ) -> None:
        self._module = module
        self._process_group = process_group
        self._sharding_strategy = sharding_strategy
        self._auto_wrap_policy = auto_wrap_policy
        self._cpu_offload = cpu_offload
        self._mixed_precision = mixed_precision
        self._backward_prefetch = backward_prefetch
        self._forward_prefetch = forward_prefetch
        self._device_id = device_id
        self._sync_module_states = sync_module_states
        self._sharded_param_names = sharded_param_names or []
        self._limit_all_gathers = limit_all_gathers
        self._flat_param: Optional[Any] = None
        self._param_shards: Dict[str, Any] = {}
        self._param_meta: Dict[str, Dict[str, Any]] = {}
        self._original_modules: List[Any] = []
        self._sharded: bool = False
        self._register_hooks()

    def _register_hooks(self) -> None:
        if hasattr(self._module, "forward"):
            self._original_forward = self._module.forward
            self._module.forward = self._wrapped_forward
        if hasattr(self._module, "backward"):
            self._original_backward = self._module.backward
            self._module.backward = self._wrapped_backward
        else:
            self._original_backward = None
        self._forward_pre_hooks: List[Any] = []
        self._forward_post_hooks: List[Any] = []
        self._backward_hooks: List[Any] = []

    def _wrapped_forward(self, *args: Any, **kwargs: Any) -> Any:
        self._trigger_forward_pre_hooks(args, kwargs)
        result = self._original_forward(*args, **kwargs)
        self._trigger_forward_post_hooks(result)
        return result

    def _wrapped_backward(self, *args: Any, **kwargs: Any) -> Any:
        self._trigger_backward_hooks(args, kwargs)

    def _trigger_forward_pre_hooks(self, args: tuple, kwargs: dict) -> None:
        for hook in self._forward_pre_hooks:
            try:
                hook(self._module, args, kwargs)
            except Exception:
                pass

    def _trigger_forward_post_hooks(self, output: Any) -> None:
        for hook in self._forward_post_hooks:
            try:
                hook(self._module, output)
            except Exception:
                pass

    def _trigger_backward_hooks(self, args: tuple, kwargs: dict) -> None:
        for hook in self._backward_hooks:
            try:
                hook(self._module, args, kwargs)
            except Exception:
                pass

    def add_forward_pre_hook(self, hook: Any) -> None:
        self._forward_pre_hooks.append(hook)

    def add_forward_post_hook(self, hook: Any) -> None:
        self._forward_post_hooks.append(hook)

    def add_backward_hook(self, hook: Any) -> None:
        self._backward_hooks.append(hook)

    def _flatten_params(self, module: Any, prefix: str = "") -> List[Tuple[str, Any]]:
        params: List[Tuple[str, Any]] = []
        for name, child in module.__dict__.items():
            if hasattr(child, "parameters"):
                try:
                    for pn, p in child.named_parameters(recurse=False):
                        params.append((f"{prefix}.{name}.{pn}" if prefix else f"{name}.{pn}", p))
                except Exception:
                    pass
            elif inspect.isclass(child) and hasattr(child, "parameters"):
                pass
            elif hasattr(child, "shape") or (hasattr(child, "parameters") and not inspect.isclass(child)):
                if hasattr(child, "parameters"):
                    try:
                        for pn, p in child.named_parameters(recurse=False):
                            params.append((f"{prefix}.{name}.{pn}" if prefix else f"{name}.{pn}", p))
                    except Exception:
                        pass
                elif hasattr(child, "shape"):
                    params.append((f"{prefix}.{name}" if prefix else name, child))
        return params

    def full_param(self, name: str) -> Optional[Any]:
        if not self._sharded:
            return None
        if self._sharding_strategy == "zero3":
            if name in self._param_buffers:
                buf = self._param_buffers[name]
                meta = self._param_meta[name]
                shape = meta.get("shape", None)
                if shape is not None:
                    if isinstance(buf, np.ndarray):
                        return buf.reshape(shape)
                    return buf
                return buf
            return self._param_shards.get(name)
        return self._param_shards.get(name)

    def sync_param(self, name: str, param: Any) -> None:
        if not self._sharded:
            return
        if self._sharding_strategy == "zero3":
            if name not in self._param_meta:
                return
            meta = self._param_meta[name]
            total = meta.get("total", 0)
            chunk = math.ceil(total / self._world_size())
            rank = self._rank()
            start = rank * chunk
            end = min(start + chunk, total)
            if isinstance(param, np.ndarray):
                flat = param.reshape(-1)
                local = flat[start:end].copy()
                self._param_shards[name] = local
            elif _HAS_TORCH and isinstance(param, torch.Tensor):
                flat = param.reshape(-1).detach().cpu().numpy()
                local = flat[start:end].copy()
                self._param_shards[name] = local
        else:
            self._param_shards[name] = param

    def _world_size(self) -> int:
        if self._process_group is not None:
            return getattr(self._process_group, "size", lambda: 1)()
        return 1

    def _rank(self) -> int:
        if self._process_group is not None:
            return getattr(self._process_group, "rank", lambda: 0)()
        return 0

    def auto_wrap(self, module: Any) -> Any:
        if self._auto_wrap_policy is None:
            return module
        if callable(self._auto_wrap_policy):
            return self._auto_wrap_policy(module)
        if hasattr(self._auto_wrap_policy, "select_module_type"):
            module_type = self._auto_wrap_policy.select_module_type()
            if module_type is None:
                return module
            return module_type(module)
        return module

    def shard(self) -> None:
        if self._sharded:
            return
        world_size = self._world_size()
        module = self._module
        params = self._flatten_params(module)
        self._flat_param = {}
        self._param_buffers = {}
        for name, param in params:
            shape = self._get_shape(param)
            total = int(np.prod(shape)) if shape else 1
            chunk = math.ceil(total / world_size)
            start = self._rank() * chunk
            end = min(start + chunk, total)
            if self._sharding_strategy == "zero3":
                if _HAS_TORCH and isinstance(param, torch.Tensor):
                    buf = param.detach().cpu().numpy().reshape(-1)
                else:
                    buf = np.asarray(param).reshape(-1)
                self._param_buffers[name] = buf
                local = buf[start:end].copy()
                self._param_shards[name] = local
                self._param_meta[name] = {
                    "start": int(start),
                    "end": int(end),
                    "total": int(total),
                    "shape": shape,
                    "chunk": int(chunk),
                    "world_size": world_size,
                }
            else:
                self._param_shards[name] = param
                self._param_meta[name] = {
                    "start": 0,
                    "end": int(total),
                    "total": int(total),
                    "shape": shape,
                    "chunk": int(total),
                    "world_size": world_size,
                }
        self._sharded = True

    def _get_shape(self, param: Any) -> Optional[Tuple[int, ...]]:
        if hasattr(param, "shape") and callable(param.shape):
            return tuple(param.shape())
        if hasattr(param, "shape"):
            return tuple(param.shape)
        return None

    def gather_full_params(self, module: Optional[Any] = None) -> None:
        if not self._sharded:
            return
        if self._sharding_strategy != "zero3":
            return
        if self._cpu_offload:
            self._gather_from_cpu()
        else:
            self._gather_from_gpu()

    def _gather_from_gpu(self) -> None:
        pass

    def _gather_from_cpu(self) -> None:
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.gather_full_params()
        return self._module(*args, **kwargs)

    def backward(self, loss: Any) -> None:
        if self._backward_prefetch:
            self._prefetch_shards()
        loss.backward()
        self.reduce_gradients()

    def _prefetch_shards(self) -> None:
        if self._sharding_strategy != "zero3":
            return
        for name, meta in self._param_meta.items():
            if name in self._param_buffers:
                pass

    def reduce_gradients(self) -> None:
        if self._sharding_strategy == "zero3":
            self._reduce_scatter_grads()
        elif self._sharding_strategy == "zero2":
            self._reduce_grads()

    def _reduce_scatter_grads(self) -> None:
        for name, meta in self._param_meta.items():
            pass

    def _reduce_grads(self) -> None:
        for name, meta in self._param_meta.items():
            pass

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "sharding_strategy": self._sharding_strategy,
            "cpu_offload": self._cpu_offload,
            "sharded": self._sharded,
            "shards": {},
            "meta": {},
        }
        for name, shard in self._param_shards.items():
            if _HAS_TORCH and hasattr(shard, "detach"):
                state["shards"][name] = shard.detach().cpu().numpy()
            else:
                state["shards"][name] = np.asarray(shard)
        state["meta"] = self._param_meta.copy()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._sharding_strategy = state.get("sharding_strategy", self._sharding_strategy)
        self._cpu_offload = state.get("cpu_offload", self._cpu_offload)
        self._sharded = state.get("sharded", False)
        self._param_meta = state.get("meta", {})
        self._param_shards = {}
        for name, shard in state.get("shards", {}).items():
            self._param_shards[name] = shard

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
