"""Auto-scaling training for any hardware cluster."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


__all__ = ["AutoScaler", "ResourceTier", "TrainingConfig"]


class ResourceTier:
    def __init__(
        self,
        name: str,
        cpu_count: int = 1,
        ram_gb: float = 4.0,
        gpu_count: int = 0,
        gpu_vram_gb: float = 0.0,
        disk_free_gb: float = 16.0,
        max_model_params: int = 10_000_000,
        recommended_batch_size: int = 32,
        gradient_accumulation: int = 1,
        parallelism: str = "none",
    ) -> None:
        self.name = name
        self.cpu_count = cpu_count
        self.ram_gb = ram_gb
        self.gpu_count = gpu_count
        self.gpu_vram_gb = gpu_vram_gb
        self.disk_free_gb = disk_free_gb
        self.max_model_params = max_model_params
        self.recommended_batch_size = recommended_batch_size
        self.gradient_accumulation = gradient_accumulation
        self.parallelism = parallelism


_TIERS: Dict[str, ResourceTier] = {
    "edge": ResourceTier("edge", cpu_count=4, ram_gb=4, max_model_params=1_000_000,
                          recommended_batch_size=8, gradient_accumulation=4, parallelism="none"),
    "laptop": ResourceTier("laptop", cpu_count=8, ram_gb=16, gpu_count=0,
                           max_model_params=10_000_000, recommended_batch_size=16,
                           gradient_accumulation=2, parallelism="cpu_offload"),
    "desktop": ResourceTier("desktop", cpu_count=16, ram_gb=32, gpu_count=1, gpu_vram_gb=8,
                            max_model_params=100_000_000, recommended_batch_size=32,
                            gradient_accumulation=1, parallelism="dp"),
    "gpu_single": ResourceTier("gpu_single", cpu_count=16, ram_gb=32, gpu_count=1, gpu_vram_gb=24,
                               max_model_params=500_000_000, recommended_batch_size=64,
                               gradient_accumulation=1, parallelism="dp"),
    "gpu_multi": ResourceTier("gpu_multi", cpu_count=32, ram_gb=64, gpu_count=4, gpu_vram_gb=24 * 4,
                              max_model_params=2_000_000_000, recommended_batch_size=128,
                              gradient_accumulation=1, parallelism="ddp"),
    "gpu_node": ResourceTier("gpu_node", cpu_count=64, ram_gb=128, gpu_count=8, gpu_vram_gb=24 * 8,
                             max_model_params=10_000_000_000, recommended_batch_size=256,
                             gradient_accumulation=1, parallelism="ddp+fsdp"),
    "cluster_small": ResourceTier("cluster_small", cpu_count=128, ram_gb=256, gpu_count=16,
                                  gpu_vram_gb=24 * 16, max_model_params=100_000_000_000,
                                  recommended_batch_size=512, gradient_accumulation=1,
                                  parallelism="fsdp+tp"),
    "cluster_medium": ResourceTier("cluster_medium", cpu_count=256, ram_gb=512, gpu_count=64,
                                   gpu_vram_gb=24 * 64, max_model_params=500_000_000_000,
                                   recommended_batch_size=1024, gradient_accumulation=1,
                                   parallelism="fsdp+tp+pp"),
    "cluster_large": ResourceTier("cluster_large", cpu_count=512, ram_gb=1024, gpu_count=128,
                                  gpu_vram_gb=24 * 128, max_model_params=2_000_000_000_000,
                                  recommended_batch_size=2048, gradient_accumulation=1,
                                  parallelism="3d+fsdp+tp+pp"),
    "supercomputer": ResourceTier("supercomputer", cpu_count=1024, ram_gb=2048, gpu_count=1000,
                                  gpu_vram_gb=24 * 1000, max_model_params=5_000_000_000_000,
                                  recommended_batch_size=4096, gradient_accumulation=1,
                                  parallelism="3d+fsdp+tp+pp"),
}


class TrainingConfig:
    def __init__(
        self,
        tier: ResourceTier,
        model_params: int,
        batch_size: int = 32,
        gradient_accumulation: int = 1,
        parallelism: str = "none",
        offload: bool = False,
        checkpoint_every: int = 1000,
        mixed_precision: bool = False,
        optimizer: str = "adam",
        lr: float = 1e-3,
        warmup_steps: int = 100,
        max_steps: int = 10000,
    ) -> None:
        self.tier = tier
        self.model_params = model_params
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.parallelism = parallelism
        self.offload = offload
        self.checkpoint_every = checkpoint_every
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.name,
            "model_params": self.model_params,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "parallelism": self.parallelism,
            "offload": self.offload,
            "checkpoint_every": self.checkpoint_every,
            "mixed_precision": self.mixed_precision,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }


class AutoScaler:
    def __init__(self, hardware_info: Optional[Any] = None) -> None:
        self.hardware_info = hardware_info
        self._tier: Optional[ResourceTier] = None
        self._detect()

    def _detect(self) -> None:
        if self.hardware_info is None:
            from aweai.hardware import detect, tier_of
            hw = detect()
            tier_name = tier_of(hw)
            self._tier = _TIERS.get(tier_name, _TIERS["laptop"])
            self._apply_hardware(hw)
        else:
            hw = self.hardware_info
            tier_name = getattr(hw, "tier", "laptop")
            self._tier = _TIERS.get(tier_name, _TIERS["laptop"])
            self._apply_hardware(hw)

    def _apply_hardware(self, hw: Any) -> None:
        gpu_count = getattr(hw, "gpu_count", 0) or 0
        gpu_vram = sum(getattr(hw, "gpu_vram_gb", [0.0]) or [0.0])
        ram = getattr(hw, "ram_total_gb", 4.0) or 4.0
        cpu = getattr(hw, "cpu_count", 1) or 1
        if gpu_count >= 128:
            self._tier = _TIERS["supercomputer"]
        elif gpu_count >= 64:
            self._tier = _TIERS["cluster_medium"]
        elif gpu_count >= 16:
            self._tier = _TIERS["cluster_small"]
        elif gpu_count >= 8:
            self._tier = _TIERS["gpu_node"]
        elif gpu_count >= 4:
            self._tier = _TIERS["gpu_multi"]
        elif gpu_count >= 1:
            self._tier = _TIERS["gpu_single"]
        elif ram >= 64:
            self._tier = _TIERS["desktop"]
        elif ram >= 16:
            self._tier = _TIERS["laptop"]
        else:
            self._tier = _TIERS["edge"]

    def detect_tier(self) -> ResourceTier:
        return self._tier

    def recommend_strategy(self, model_params: int) -> Dict[str, Any]:
        tier = self._tier
        recommended_parallelism = tier.parallelism
        offload = False
        if model_params > tier.max_model_params:
            offload = True
            if "tp" not in recommended_parallelism:
                recommended_parallelism += "+tp"
            if "pp" not in recommended_parallelism and model_params > tier.max_model_params * 10:
                recommended_parallelism += "+pp"
        batch_size = tier.recommended_batch_size
        if model_params > tier.max_model_params * 100:
            batch_size = max(1, batch_size // 4)
        elif model_params > tier.max_model_params * 10:
            batch_size = max(1, batch_size // 2)
        grad_accum = tier.gradient_accumulation
        if model_params > tier.max_model_params:
            grad_accum = max(grad_accum, 2)
        return {
            "tier": tier.name,
            "parallelism": recommended_parallelism,
            "batch_size": batch_size,
            "gradient_accumulation": grad_accum,
            "offload": offload,
            "max_model_params_for_tier": tier.max_model_params,
            "model_params": model_params,
        }

    def build_config(
        self,
        model_params: int,
        max_steps: int = 10000,
        lr: Optional[float] = None,
        optimizer: Optional[str] = None,
    ) -> TrainingConfig:
        tier = self._tier
        strategy = self.recommend_strategy(model_params)
        batch_size = strategy["batch_size"]
        grad_accum = strategy["gradient_accumulation"]
        return TrainingConfig(
            tier=tier,
            model_params=model_params,
            batch_size=batch_size,
            gradient_accumulation=grad_accum,
            parallelism=strategy["parallelism"],
            offload=strategy["offload"],
            checkpoint_every=max(100, min(1000, max_steps // 10)),
            mixed_precision=_HAS_TORCH and tier.gpu_count > 0,
            optimizer=optimizer or "adam",
            lr=lr or 1e-3,
            warmup_steps=max(10, max_steps // 100),
            max_steps=max_steps,
        )

    def adaptive_batch_size(
        self,
        initial_batch_size: int,
        model_params: int,
        step_loss: float,
        oom_events: int = 0,
    ) -> int:
        tier = self._tier
        batch_size = initial_batch_size
        if oom_events > 0:
            batch_size = max(1, batch_size // 2)
        if step_loss > 100.0 and batch_size > 1:
            batch_size = max(1, batch_size // 2)
        if step_loss < 0.1 and batch_size < tier.recommended_batch_size:
            batch_size = min(tier.recommended_batch_size, batch_size * 2)
        if model_params > tier.max_model_params and batch_size > 1:
            batch_size = max(1, batch_size // 2)
        return batch_size

    def elasticity_score(self, world_size: int, model_params: int) -> float:
        tier = self._tier
        capacity = tier.max_model_params * max(1, world_size)
        if model_params <= 0:
            return 1.0
        return min(1.0, capacity / model_params)

    def recommend_world_size(self, model_params: int) -> int:
        tier = self._tier
        min_needed = math.ceil(model_params / max(1, tier.max_model_params))
        gpu_available = max(tier.gpu_count, 1)
        return min(gpu_available, max(1, min_needed))

    def should_use_checkpointing(self, model_params: int) -> bool:
        tier = self._tier
        return model_params > tier.max_model_params * 0.5
