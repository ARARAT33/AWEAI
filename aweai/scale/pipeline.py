"""Pipeline Parallelism implementation."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


__all__ = ["PipelineStage", "PipelineParallel"]


class PipelineStage:
    def __init__(
        self,
        stage_id: int,
        submodule: Any,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        device: str = "cpu",
        micro_batch_size: int = 1,
    ) -> None:
        self.stage_id = stage_id
        self.submodule = submodule
        self.input_names = input_names or ["hidden"]
        self.output_names = output_names or ["hidden"]
        self.device = device
        self.micro_batch_size = micro_batch_size
        self._weight: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._activation_cache: List[Any] = []
        self._grad_cache: List[Any] = []
        self._init_parameters()

    def _init_parameters(self) -> None:
        sub = self.submodule
        if hasattr(sub, "get_param_names"):
            for pname in sub.get_param_names():
                pass
        elif hasattr(sub, "state_dict"):
            try:
                sd = sub.state_dict()
                for k, v in sd.items():
                    if hasattr(v, "shape"):
                        pass
            except Exception:
                pass
        elif hasattr(sub, "W") or hasattr(sub, "weight"):
            W = getattr(sub, "W", None) or getattr(sub, "weight", None)
            if W is not None:
                if _HAS_TORCH and hasattr(W, "detach"):
                    W = W.detach().cpu().numpy()
                if hasattr(W, "shape"):
                    self._weight = np.asarray(W)

    def get_param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {}
        if self._weight is not None:
            shapes["weight"] = tuple(self._weight.shape)
        if self._bias is not None:
            shapes["bias"] = tuple(self._bias.shape)
        return shapes

    def forward(self, microbatch: Any) -> Any:
        if hasattr(self.submodule, "forward"):
            return self.submodule.forward(microbatch)
        if hasattr(self.submodule, "__call__"):
            return self.submodule(microbatch)
        if self._weight is not None:
            x = np.asarray(microbatch, dtype=np.float32)
            out = np.matmul(x, self._weight.T)
            if self._bias is not None:
                out = out + self._bias
            return out
        return np.asarray(microbatch)

    def backward(self, upstream: Any) -> Any:
        if hasattr(self.submodule, "backward"):
            return self.submodule.backward(upstream)
        if hasattr(self.submodule, "backward"):
            return self.submodule.backward(upstream)
        return upstream

    def step(self, optimizer: Any) -> None:
        if optimizer is not None and hasattr(optimizer, "step"):
            optimizer.step()

    def zero_grad(self) -> None:
        if self._weight is not None:
            pass
        if hasattr(self.submodule, "zero_grad"):
            self.submodule.zero_grad()

    def parameters(self) -> List[Any]:
        params: List[Any] = []
        if self._weight is not None:
            params.append(self._weight)
        if self._bias is not None:
            params.append(self._bias)
        if hasattr(self.submodule, "parameters"):
            try:
                params.extend(list(self.submodule.parameters()))
            except Exception:
                pass
        return params


class PipelineParallel:
    def __init__(
        self,
        stages: List[PipelineStage],
        chunk_mode: str = "uniform",
        schedule: str = "1f1b",
        num_microbatches: int = 4,
        grad_accumulation_steps: int = 1,
        device_map: Optional[Dict[int, str]] = None,
        grad_sync_after_forward: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("stages must be non-empty")
        self.stages = stages
        self.chunk_mode = chunk_mode
        self.schedule = schedule
        self.num_microbatches = num_microbatches
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device_map = device_map or {i: s.device for i, s in enumerate(stages)}
        self.grad_sync_after_forward = grad_sync_after_forward
        self._microbatches: List[Any] = []
        self._bubble_count: int = 0
        self._tick_count: int = 0
        self._stage_outputs: Dict[int, deque] = {i: deque() for i in range(len(stages))}
        self._gradients: Dict[int, deque] = {i: deque() for i in range(len(stages))}
        self._forward_done: List[bool] = [False] * len(stages)
        self._backward_done: List[bool] = [False] * len(stages)
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(stages)))

    def set_microbatches(self, microbatch_list: List[Any]) -> None:
        self._microbatches = list(microbatch_list)

    def _split_microbatches(self, batch: Any) -> List[Any]:
        if not isinstance(batch, np.ndarray):
            return [batch for _ in range(self.num_microbatches)]
        n = batch.shape[0]
        chunk = max(1, n // self.num_microbatches)
        mb_list = []
        for i in range(self.num_microbatches):
            lo = i * chunk
            hi = n if i == self.num_microbatches - 1 else (i + 1) * chunk
            mb_list.append(batch[lo:hi])
        return mb_list

    def _assign_stages(self) -> List[int]:
        if self.chunk_mode == "uniform":
            n = len(self.stages)
            base = self.num_microbatches // n
            rem = self.num_microbatches % n
            counts = [base + (1 if i < rem else 0) for i in range(n)]
        elif self.chunk_mode == "balanced":
            counts = self._balanced_assign()
        else:
            counts = [self.num_microbatches // len(self.stages)] * len(self.stages)
            rem = self.num_microbatches % len(self.stages)
            for i in range(rem):
                counts[i] += 1
        return counts

    def _balanced_assign(self) -> List[int]:
        n = len(self.stages)
        counts = [0] * n
        for i in range(self.num_microbatches):
            idx = i % n
            counts[idx] += 1
        return counts

    def forward(self, batch: Any) -> Any:
        if not self._microbatches:
            self._microbatches = self._split_microbatches(batch)
        counts = self._assign_stages()
        stage_assignments: List[List[int]] = [[] for _ in range(len(self.stages))]
        mb_idx = 0
        for stage_idx, count in enumerate(counts):
            for _ in range(count):
                if mb_idx < len(self._microbatches):
                    stage_assignments[stage_idx].append(mb_idx)
                    mb_idx += 1
        outputs = []
        if self.schedule == "1f1b":
            outputs = self._schedule_1f1b(stage_assignments)
        elif self.schedule == "gpipe":
            outputs = self._schedule_gpipe(stage_assignments)
        elif self.schedule == "interleaved":
            outputs = self._schedule_interleaved(stage_assignments)
        else:
            outputs = self._schedule_1f1b(stage_assignments)
        return self._aggregate_outputs(outputs)

    def _schedule_1f1b(self, stage_assignments: List[List[int]]) -> List[Any]:
        num_stages = len(self.stages)
        max_mb = max((len(a) for a in stage_assignments), default=0)
        outputs = []
        warmup_steps = min(num_stages - 1, max_mb)
        for mb_id in range(max_mb):
            self._tick_count += 1
            current_stage = mb_id % num_stages
            assigned = stage_assignments[current_stage]
            if mb_id < len(assigned):
                mb_idx = assigned[mb_id]
                output = self._forward_stage(current_stage, self._microbatches[mb_idx])
                outputs.append(output)
        return outputs

    def _schedule_gpipe(self, stage_assignments: List[List[int]]) -> List[Any]:
        num_stages = len(self.stages)
        max_mb = max((len(a) for a in stage_assignments), default=0)
        outputs = []
        for mb_id in range(max_mb):
            self._tick_count += 1
            for stage_idx in range(num_stages):
                assigned = stage_assignments[stage_idx]
                if mb_id < len(assigned):
                    mb_idx = assigned[mb_id]
                    output = self._forward_stage(stage_idx, self._microbatches[mb_idx])
                    outputs.append(output)
        return outputs

    def _schedule_interleaved(self, stage_assignments: List[List[int]]) -> List[Any]:
        num_stages = len(self.stages)
        max_mb = max((len(a) for a in stage_assignments), default=0)
        outputs = []
        for step in range(max_mb):
            self._tick_count += 1
            for stage_idx in range(num_stages):
                assigned = stage_assignments[stage_idx]
                if step < len(assigned):
                    mb_idx = assigned[step]
                    output = self._forward_stage(stage_idx, self._microbatches[mb_idx])
                    outputs.append(output)
        return outputs

    def _forward_stage(self, stage_idx: int, microbatch: Any) -> Any:
        stage = self.stages[stage_idx]
        return stage.forward(microbatch)

    def _backward_stage(self, stage_idx: int, upstream: Any) -> Any:
        stage = self.stages[stage_idx]
        return stage.backward(upstream)

    def backward(self, loss: Any) -> None:
        num_stages = len(self.stages)
        for stage_idx in reversed(range(num_stages)):
            if self._gradients[stage_idx]:
                upstream = self._gradients[stage_idx].pop()
                self._backward_stage(stage_idx, upstream)

    def step(self, optimizer: Any) -> None:
        for stage in self.stages:
            stage.step(optimizer)
        self._bubble_count = 0

    def zero_grad(self) -> None:
        for stage in self.stages:
            stage.zero_grad()

    def _aggregate_outputs(self, outputs: List[Any]) -> Any:
        if not outputs:
            return None
        if all(isinstance(o, np.ndarray) for o in outputs):
            return np.concatenate(outputs, axis=0)
        return outputs[-1]

    def compute_bubble_fraction(self) -> float:
        total_ticks = self._tick_count
        if total_ticks == 0:
            return 0.0
        return self._bubble_count / total_ticks

    def stage_assignments(self) -> Dict[int, List[int]]:
        counts = self._assign_stages()
        assignments: Dict[int, List[int]] = {}
        mb_idx = 0
        for stage_idx, count in enumerate(counts):
            assignments[stage_idx] = list(range(mb_idx, mb_idx + count))
            mb_idx += count
        return assignments

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
