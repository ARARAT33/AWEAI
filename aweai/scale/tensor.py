"""Tensor Parallelism implementation with 1D/2D/2.5D strategies."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


__all__ = ["TensorParallelLinear", "TensorParallelMLP", "TensorParallelStrategy"]


class TensorParallelStrategy:
    def __init__(
        self,
        mode: str = "1d",
        tp_size: int = 1,
        group_size: int = 1,
        dim: int = -1,
    ) -> None:
        if mode not in ("1d", "2d", "2.5d"):
            raise ValueError("mode must be '1d', '2d', or '2.5d'")
        self.mode = mode
        self.tp_size = tp_size
        self.group_size = group_size
        self.dim = dim
        self._row_rank: int = 0
        self._col_rank: int = 0
        self._mesh_shape: Tuple[int, int] = (1, 1)
        self._setup_mesh()

    def _setup_mesh(self) -> None:
        if self.mode == "1d":
            self._mesh_shape = (self.tp_size, 1)
            self._col_rank = 0
            self._row_rank = 0
        elif self.mode == "2d":
            sqrt = int(math.sqrt(self.tp_size))
            if sqrt * sqrt != self.tp_size:
                sqrt = int(math.ceil(math.sqrt(self.tp_size)))
                while sqrt * (sqrt - 1) >= self.tp_size:
                    sqrt -= 1
                if sqrt * (sqrt - 1) < self.tp_size:
                    sqrt += 1
            row = sqrt
            col = math.ceil(self.tp_size / row)
            self._mesh_shape = (row, col)
        elif self.mode == "2.5d":
            self._mesh_shape = (2, max(1, self.tp_size // 2))

    def row_rank(self) -> int:
        return self._row_rank

    def col_rank(self) -> int:
        return self._col_rank

    def mesh_shape(self) -> Tuple[int, int]:
        return self._mesh_shape

    def shard_weight(self, weight: Any, shard_type: str = "column") -> Any:
        if self.tp_size <= 1:
            return weight
        arr = self._to_np(weight)
        if shard_type == "column":
            return self._shard_column(arr)
        if shard_type == "row":
            return self._shard_row(arr)
        return self._shard_row(arr)

    def _shard_column(self, arr: np.ndarray) -> np.ndarray:
        dim = self.dim if self.dim >= 0 else len(arr.shape) - 1
        total = arr.shape[dim]
        chunk = math.ceil(total / self.tp_size)
        start = (0 if self.mode == "1d" else self._col_rank) * chunk
        end = min(start + chunk, total)
        sl = [slice(None)] * len(arr.shape)
        sl[dim] = slice(start, end)
        return arr[tuple(sl)]

    def _shard_row(self, arr: np.ndarray) -> np.ndarray:
        dim = 0 if self.dim < 0 else self.dim
        if dim >= len(arr.shape):
            dim = 0
        total = arr.shape[dim]
        chunk = math.ceil(total / self.tp_size)
        start = (0 if self.mode == "1d" else self._row_rank) * chunk
        end = min(start + chunk, total)
        sl = [slice(None)] * len(arr.shape)
        sl[dim] = slice(start, end)
        return arr[tuple(sl)]

    def all_gather(self, shard: Any, dim: int = -1) -> np.ndarray:
        if self.tp_size <= 1:
            return self._to_np(shard)
        arr = self._to_np(shard)
        if self.mode == "1d":
            return self._all_gather_1d(arr, dim)
        if self.mode == "2d":
            return self._all_gather_2d(arr, dim)
        return self._all_gather_2d(arr, dim)

    def _all_gather_1d(self, arr: np.ndarray, dim: int) -> np.ndarray:
        dim = dim if dim >= 0 else len(arr.shape) - 1
        shards = [arr.copy() for _ in range(self.tp_size)]
        for i, shard in enumerate(shards):
            total = getattr(arr, "_tp_total", arr.shape[dim])
            chunk = math.ceil(total / self.tp_size)
            start = i * chunk
            end = min(start + chunk, total)
            sl = [slice(None)] * len(arr.shape)
            sl[dim] = slice(start, end)
            shards[i] = arr[tuple(sl)]
        return np.concatenate(shards, axis=dim)

    def _all_gather_2d(self, arr: np.ndarray, dim: int) -> np.ndarray:
        dim = dim if dim >= 0 else len(arr.shape) - 1
        row, col = self._mesh_shape
        shards = []
        for r in range(row):
            for c in range(col):
                if r == self._row_rank and c == self._col_rank:
                    shards.append(arr.copy())
                else:
                    shape = list(arr.shape)
                    shape[dim] = 1
                    shards.append(np.zeros(shape, dtype=arr.dtype))
        return np.concatenate(shards, axis=dim)

    def reduce_scatter(self, arr: Any, dim: int = -1) -> np.ndarray:
        if self.tp_size <= 1:
            return self._to_np(arr)
        full = self._to_np(arr)
        dim = dim if dim >= 0 else len(full.shape) - 1
        total = full.shape[dim]
        chunk = math.ceil(total / self.tp_size)
        start = (0 if self.mode == "1d" else self._col_rank) * chunk
        end = min(start + chunk, total)
        sl = [slice(None)] * len(full.shape)
        sl[dim] = slice(start, end)
        return full[tuple(sl)]

    def _to_np(self, obj: Any) -> np.ndarray:
        if isinstance(obj, np.ndarray):
            return obj
        if _HAS_TORCH and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return np.asarray(obj)

    def _to_torch(self, obj: Any) -> Any:
        if not _HAS_TORCH:
            return obj
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        return torch.tensor(obj)


class TensorParallelLinear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_strategy: Optional[TensorParallelStrategy] = None,
        shard_type: str = "column",
        device: str = "cpu",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.tp_strategy = tp_strategy or TensorParallelStrategy(mode="1d", tp_size=1)
        self.shard_type = shard_type
        self.device = device
        self._bias = np.zeros(out_features, dtype=np.float32) if bias else None
        if shard_type == "column":
            self._local_out = max(1, math.ceil(out_features / self.tp_strategy.tp_size))
            self._weight = np.random.randn(self._local_out, in_features).astype(np.float32) * 0.1
        else:
            self._local_in = max(1, math.ceil(in_features / self.tp_strategy.tp_size))
            self._weight = np.random.randn(out_features, self._local_in).astype(np.float32) * 0.1
        self._gather_output = shard_type == "column"

    def parameters(self) -> List[np.ndarray]:
        params = [self._weight]
        if self._bias is not None:
            params.append(self._bias)
        return params

    def forward(self, x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = np.matmul(x, self._weight.T)
        if self._bias is not None and self.shard_type == "column":
            out = out + self._bias
        if self._gather_output and self.tp_strategy.tp_size > 1:
            gathered = self.tp_strategy.all_gather(out, dim=-1)
            if self._bias is not None:
                gathered = gathered + self._bias
            return gathered
        return out

    def backward(self, grad: Any) -> np.ndarray:
        return np.asarray(grad)

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "weight": self._weight,
            "bias": self._bias,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "shard_type": self.shard_type,
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._weight = np.asarray(state.get("weight", self._weight))
        self._bias = state.get("bias", self._bias)
        if self._bias is not None:
            self._bias = np.asarray(self._bias)
        self.in_features = int(state.get("in_features", self.in_features))
        self.out_features = int(state.get("out_features", self.out_features))
        self.shard_type = str(state.get("shard_type", self.shard_type))


class TensorParallelMLP:
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        tp_strategy: Optional[TensorParallelStrategy] = None,
        device: str = "cpu",
        activation: str = "gelu",
    ) -> None:
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.device = device
        self.activation = activation
        self.tp_strategy = tp_strategy or TensorParallelStrategy(mode="1d", tp_size=1)
        self.fc1 = TensorParallelLinear(
            in_features,
            hidden_features,
            bias=True,
            tp_strategy=self.tp_strategy,
            shard_type="column",
            device=device,
        )
        self.fc2 = TensorParallelLinear(
            hidden_features,
            out_features,
            bias=True,
            tp_strategy=self.tp_strategy,
            shard_type="row",
            device=device,
        )

    def parameters(self) -> List[np.ndarray]:
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x: Any) -> np.ndarray:
        x = self.fc1.forward(x)
        if self.activation == "gelu":
            x = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        elif self.activation == "relu":
            x = np.maximum(0, x)
        elif self.activation == "silu":
            x = x * (1.0 / (1.0 + np.exp(-x)))
        else:
            x = np.maximum(0, x)
        x = self.fc2.forward(x)
        return x

    def backward(self, grad: Any) -> np.ndarray:
        return np.asarray(grad)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "fc1" in state:
            self.fc1.load_state_dict(state["fc1"])
        if "fc2" in state:
            self.fc2.load_state_dict(state["fc2"])
