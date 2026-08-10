from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class StackedModel(nn.Module):
    def __init__(self, models: List[nn.Module], mode: str = "sequential") -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sequential":
            for model in self.models:
                x = model(x)
            return x
        elif self.mode == "residual":
            out = x
            for model in self.models:
                out = model(out)
            return x + out
        else:
            raise ValueError(f"Unknown stacking mode: {self.mode}")


class CascadeModel(nn.Module):
    def __init__(self, stages: List[nn.Module], threshold: float = 0.9) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for stage in self.stages[:-1]:
            out = stage(out)
            conf = out.abs().mean(dim=-1)
            if conf.min() > self.threshold:
                return out
        return self.stages[-1](out)


class EnsembleRouter(nn.Module):
    def __init__(self, experts: List[nn.Module], d_model: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(d_model, len(experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + weights[..., i : i + 1] * expert(x)
        return out


class ArchitectureCompound(nn.Module):
    def __init__(self, components: List[nn.Module], connections: List[tuple[int, int]], d_model: int) -> None:
        super().__init__()
        self.components = nn.ModuleList(components)
        self.connections = connections
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for i, comp in enumerate(self.components):
            inp = torch.cat([outputs[src] for src, dst in self.connections if dst == i], dim=-1)
            if inp.size(-1) != self.d_model:
                inp = inp[..., : self.d_model]
            outputs.append(comp(inp))
        return outputs[-1]


class HierarchicalMoE(nn.Module):
    def __init__(self, d_model: int, num_experts: int, num_levels: int, expert_hidden: int) -> None:
        super().__init__()
        from .moe import MoELayer
        self.levels = nn.ModuleList([
            MoELayer(d_model, num_experts, expert_hidden, k=2, capacity_factor=1.0 + i * 0.25)
            for i in range(num_levels)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux = 0.0
        for level in self.levels:
            x, a = level(x)
            aux = aux + a
        return self.norm(x), aux
