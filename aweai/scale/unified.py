"""Unified training orchestrator composing ZeRO, FSDP, Pipeline, Tensor, and Offload."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from aweai.scale.zero import ZeROStage123
from aweai.scale.fsdp import FSDPWrapper
from aweai.scale.pipeline import PipelineParallel, PipelineStage
from aweai.scale.tensor import TensorParallelLinear, TensorParallelMLP, TensorParallelStrategy
from aweai.scale.offload import OffloadEngine
from aweai.scale.autoscale import AutoScaler, TrainingConfig


__all__ = ["UnlimitedTrainer", "Checkpoint"]


class Checkpoint:
    def __init__(
        self,
        step: int = 0,
        epoch: int = 0,
        state: Optional[Dict[str, Any]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.step = step
        self.epoch = epoch
        self.state = state or {}
        self.optimizer_state = optimizer_state or {}
        self.config = config or {}
        self.timestamp = time.time()

    def save(self, path: str) -> None:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.step,
            "epoch": self.epoch,
            "state": self.state,
            "optimizer_state": self.optimizer_state,
            "config": self.config,
            "timestamp": self.timestamp,
        }
        p.write_text(np.array(payload).dumps(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return cls(step=0, epoch=0, state={}, optimizer_state={}, config={})


class UnlimitedTrainer:
    def __init__(
        self,
        model: Any,
        config: Optional[TrainingConfig] = None,
        world_size: int = 1,
        rank: int = 0,
        checkpoint_dir: Optional[str] = None,
        enable_elastic: bool = False,
        logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.checkpoint_dir = checkpoint_dir or "./checkpoints"
        self.enable_elastic = enable_elastic
        self.logger = logger
        self._zero: Optional[ZeROStage123] = None
        self._fsdp: Optional[FSDPWrapper] = None
        self._pipeline: Optional[PipelineParallel] = None
        self._offload: Optional[OffloadEngine] = None
        self._tp_strategy: Optional[TensorParallelStrategy] = None
        self._step: int = 0
        self._epoch: int = 0
        self._losses: List[float] = []
        self._elastic_checkpoints: List[str] = []
        self._setup_strategies()

    def _setup_strategies(self) -> None:
        if self.config is None:
            from aweai.scale.autoscale import AutoScaler
            scaler = AutoScaler()
            self.config = scaler.build_config(model_params=0, max_steps=1000)
        parallelism = self.config.parallelism
        if "fsdp" in parallelism or self.config.offload:
            self._fsdp = FSDPWrapper(
                module=self.model,
                sharding_strategy="zero3" if "zero3" in parallelism else "zero2",
                cpu_offload=self.config.offload,
                sync_module_states=True,
            )
            self._fsdp.shard()
        if "zero" in parallelism:
            stage = 3 if "zero3" in parallelism else (2 if "zero2" in parallelism else 1)
            self._zero = ZeROStage123(
                stage=stage,
                world_size=self.world_size,
                rank=self.rank,
                cpu_offload=self.config.offload,
            )
        if "tp" in parallelism:
            tp_size = self._infer_tp_size(parallelism)
            self._tp_strategy = TensorParallelStrategy(mode="1d", tp_size=tp_size)
        if "pp" in parallelism:
            pp_size = self._infer_pp_size(parallelism)
            stages = self._build_pipeline_stages(pp_size)
            self._pipeline = PipelineParallel(
                stages=stages,
                num_microbatches=max(1, self.config.batch_size // pp_size),
                schedule="1f1b",
            )
        if self.config.offload:
            self._offload = OffloadEngine(
                ram_limit_fraction=0.8,
                ssd_enabled=True,
                nvme_enabled=True,
                async_prefetch=True,
            )

    def _infer_tp_size(self, parallelism: str) -> int:
        if self.config and self.config.tier.gpu_count > 0:
            return min(self.config.tier.gpu_count, 8)
        return 1

    def _infer_pp_size(self, parallelism: str) -> int:
        if self.config and self.config.tier.gpu_count > 0:
            return min(max(1, self.config.tier.gpu_count // 4), 8)
        return 1

    def _build_pipeline_stages(self, pp_size: int) -> List[PipelineStage]:
        stages = []
        submodules = self._split_model(self.model, pp_size)
        for i, sub in enumerate(submodules):
            stage = PipelineStage(stage_id=i, submodule=sub, device="cpu")
            stages.append(stage)
        return stages

    def _split_model(self, model: Any, pp_size: int) -> List[Any]:
        if hasattr(model, "layers"):
            layers = list(model.layers)
            n = len(layers)
            chunk = max(1, math.ceil(n / pp_size))
            return [layers[i * chunk:min((i + 1) * chunk, n)] for i in range(pp_size)]
        return [model for _ in range(pp_size)]

    def train_step(self, batch: Any, target: Any, optimizer: Any, loss_fn: Any) -> Dict[str, Any]:
        forward_output = self.forward(batch)
        loss = self.compute_loss(forward_output, target, loss_fn)
        self.backward(loss)
        self.step(optimizer)
        metrics = {"loss": float(np.mean(np.asarray(loss))), "step": self._step}
        self._losses.append(metrics["loss"])
        self._step += 1
        return metrics

    def forward(self, batch: Any) -> Any:
        if self._pipeline is not None:
            return self._pipeline.forward(batch)
        if self._fsdp is not None:
            return self._fsdp.forward(batch)
        if hasattr(self.model, "forward"):
            return self.model.forward(batch)
        if hasattr(self.model, "__call__"):
            return self.model(batch)
        if self._tp_strategy is not None and hasattr(self.model, "forward"):
            x = np.asarray(batch, dtype=np.float32)
            out = self.model.forward(x)
            return out
        return np.asarray(batch)

    def compute_loss(self, output: Any, target: Any, loss_fn: Any) -> Any:
        if loss_fn is not None and callable(loss_fn):
            return loss_fn(output, target)
        out = np.asarray(output, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        diff = out - tgt
        return np.mean(diff ** 2)

    def backward(self, loss: Any) -> None:
        if self._pipeline is not None:
            self._pipeline.backward(loss)
        if self._fsdp is not None:
            self._fsdp.backward(loss)
            self._fsdp.reduce_gradients()
        if self._zero is not None:
            self._zero.reduce_gradients({})

    def step(self, optimizer: Any) -> None:
        if self._pipeline is not None:
            self._pipeline.step(optimizer)
        if self._fsdp is not None:
            self._fsdp.step(optimizer)
        if self._zero is not None:
            self._zero.step(optimizer)
        if self._zero is not None:
            self._zero.zero_grad({})
        if self._fsdp is not None:
            self._fsdp.zero_grad()

    def train_epoch(self, dataloader: Any, optimizer: Any, loss_fn: Any) -> Dict[str, Any]:
        epoch_losses = []
        for batch, target in dataloader:
            metrics = self.train_step(batch, target, optimizer, loss_fn)
            epoch_losses.append(metrics["loss"])
            if self._should_checkpoint():
                self.save_checkpoint()
        self._epoch += 1
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        return {"epoch": self._epoch, "avg_loss": avg_loss, "steps": len(epoch_losses)}

    def _should_checkpoint(self) -> bool:
        if self.config is None:
            return self._step % 1000 == 0
        return self._step % self.config.checkpoint_every == 0

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        ckpt_path = path or f"{self.checkpoint_dir}/ckpt_step_{self._step}.pkl"
        state = self._collect_state()
        optimizer_state = {}
        ckpt = Checkpoint(
            step=self._step,
            epoch=self._epoch,
            state=state,
            optimizer_state=optimizer_state,
            config=self.config.to_dict() if self.config else {},
        )
        ckpt.save(ckpt_path)
        self._elastic_checkpoints.append(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str) -> None:
        ckpt = Checkpoint.load(path)
        self._step = ckpt.step
        self._epoch = ckpt.epoch
        self._restore_state(ckpt.state)

    def _collect_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"model": {}, "zero": None, "fsdp": None, "offload": None}
        if hasattr(self.model, "state_dict"):
            try:
                state["model"] = self.model.state_dict()
            except Exception:
                pass
        if self._zero is not None:
            try:
                state["zero"] = self._zero.state_dict()
            except Exception:
                pass
        if self._fsdp is not None:
            try:
                state["fsdp"] = self._fsdp.state_dict()
            except Exception:
                pass
        if self._offload is not None:
            try:
                state["offload"] = self._offload.stats()
            except Exception:
                pass
        return state

    def _restore_state(self, state: Dict[str, Any]) -> None:
        if "zero" in state and state["zero"] is not None and self._zero is not None:
            self._zero.load_state_dict(state["zero"])
        if "fsdp" in state and state["fsdp"] is not None and self._fsdp is not None:
            self._fsdp.load_state_dict(state["fsdp"])
        if "model" in state and state["model"] and hasattr(self.model, "load_state_dict"):
            try:
                self.model.load_state_dict(state["model"])
            except Exception:
                pass

    def elastic_resize(self, new_world_size: int) -> Dict[str, Any]:
        if not self.enable_elastic:
            return {"status": "disabled"}
        old_world_size = self.world_size
        self.world_size = new_world_size
        if self._zero is not None:
            self._zero.world_size = new_world_size
            self._zero.rank = self.rank
        if self._fsdp is not None:
            pass
        return {
            "status": "resized",
            "old_world_size": old_world_size,
            "new_world_size": new_world_size,
        }

    def fit(
        self,
        dataloader: Any,
        epochs: int = 1,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        for epoch in range(epochs):
            epoch_metrics = self.train_epoch(dataloader, optimizer, loss_fn)
            history.append(epoch_metrics)
        return {"history": history, "final_loss": self._losses[-1] if self._losses else 0.0}

    def summary(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "epoch": self._epoch,
            "world_size": self.world_size,
            "rank": self.rank,
            "zero": self._zero.stage if self._zero else None,
            "fsdp": self._fsdp._sharding_strategy if self._fsdp else None,
            "pipeline": self._pipeline.schedule if self._pipeline else None,
            "tp": self._tp_strategy.mode if self._tp_strategy else None,
            "offload": self._offload is not None,
            "elastic": self.enable_elastic,
            "checkpoints": len(self._elastic_checkpoints),
            "last_loss": self._losses[-1] if self._losses else None,
        }
