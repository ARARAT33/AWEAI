"""Scale module for unlimited-parameter model training."""

from __future__ import annotations

from aweai.scale.zero import ZeROStage123
from aweai.scale.fsdp import FSDPWrapper
from aweai.scale.pipeline import PipelineStage, PipelineParallel
from aweai.scale.tensor import TensorParallelLinear, TensorParallelMLP
from aweai.scale.offload import OffloadEngine
from aweai.scale.autoscale import AutoScaler, ResourceTier, TrainingConfig
from aweai.scale.unified import UnlimitedTrainer, Checkpoint

__version__ = "4.3.0"

__all__ = [
    "ZeROStage123",
    "FSDPWrapper",
    "PipelineStage",
    "PipelineParallel",
    "TensorParallelLinear",
    "TensorParallelMLP",
    "OffloadEngine",
    "AutoScaler",
    "ResourceTier",
    "TrainingConfig",
    "UnlimitedTrainer",
    "Checkpoint",
]
