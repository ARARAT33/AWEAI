from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DesignSpace:
    d_model: Tuple[int, ...] = (128, 256, 512, 768, 1024)
    num_heads: Tuple[int, ...] = (4, 8, 12, 16)
    num_layers: Tuple[int, ...] = (2, 4, 6, 8, 12, 24)
    d_ff: Tuple[int, ...] = (512, 1024, 2048, 4096)
    families: Tuple[str, ...] = ("Transformer", "MoE", "NextGen")
    dropout: Tuple[float, ...] = (0.0, 0.1, 0.2)
    vocab_size: Tuple[int, ...] = (32000, 65536, 100000)


@dataclass
class HardwareTarget:
    name: str
    gpu_memory_gb: int
    max_batch: int = 8
    max_seq_len: int = 2048
    target_latency_ms: float = 100.0
    tensor_cores: bool = True


class AutoDesigner:
    def __init__(self, design_space: Optional[DesignSpace] = None, hardware: Optional[HardwareTarget] = None) -> None:
        self.design_space = design_space or DesignSpace()
        self.hardware = hardware or HardwareTarget(name="A100", gpu_memory_gb=80)
        self.population: List[Dict[str, Any]] = []
        self.generation: int = 0

    def initialize_population(self, size: int = 20) -> List[Dict[str, Any]]:
        import random
        space = self.design_space
        pop = []
        for _ in range(size):
            candidate = {
                "d_model": random.choice(space.d_model),
                "num_heads": random.choice(space.num_heads),
                "num_layers": random.choice(space.num_layers),
                "d_ff": random.choice(space.d_ff),
                "family": random.choice(space.families),
                "dropout": random.choice(space.dropout),
                "vocab_size": random.choice(space.vocab_size),
                "fitness": 0.0,
            }
            candidate["score"] = self._score(candidate)
            pop.append(candidate)
        self.population = sorted(pop, key=lambda c: c["score"], reverse=True)
        return self.population

    def evolve(self, generations: int = 10, mutate_rate: float = 0.2) -> List[Dict[str, Any]]:
        for _ in range(generations):
            self.generation += 1
            top = self.population[: max(2, len(self.population) // 4)]
            new_pop = list(top)
            while len(new_pop) < len(self.population):
                parent = random.choice(top)
                child = dict(parent)
                if random.random() < mutate_rate:
                    child["d_model"] = random.choice(self.design_space.d_model)
                if random.random() < mutate_rate:
                    child["num_layers"] = random.choice(self.design_space.num_layers)
                if random.random() < mutate_rate:
                    child["d_ff"] = random.choice(self.design_space.d_ff)
                if random.random() < mutate_rate:
                    child["family"] = random.choice(self.design_space.families)
                child["score"] = self._score(child)
                new_pop.append(child)
            self.population = sorted(new_pop, key=lambda c: c["score"], reverse=True)
        return self.population

    def _score(self, candidate: Dict[str, Any]) -> float:
        mem = candidate["d_model"] * candidate["num_layers"] * 4
        params = candidate["d_model"] * candidate["num_layers"] * candidate["d_ff"] * 2
        mem_penalty = max(0.0, (mem / self.hardware.gpu_memory_gb - 1.0)) * -1000.0
        latency_penalty = max(0.0, (params / 1e9 - 1.0)) * -100.0
        score = candidate["d_model"] * candidate["num_layers"] + mem_penalty + latency_penalty
        return score

    def best(self) -> Dict[str, Any]:
        return self.population[0] if self.population else {}

    def constraint_based(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        import random
        results = []
        for _ in range(50):
            candidate = {
                "d_model": random.choice(self.design_space.d_model),
                "num_heads": random.choice(self.design_space.num_heads),
                "num_layers": random.choice(self.design_space.num_layers),
                "d_ff": random.choice(self.design_space.d_ff),
                "family": random.choice(self.design_space.families),
                "dropout": random.choice(self.design_space.dropout),
                "vocab_size": random.choice(self.design_space.vocab_size),
            }
            if all(candidate.get(k) == v or candidate.get(k) in v for k, v in constraints.items()):
                candidate["score"] = self._score(candidate)
                results.append(candidate)
        return sorted(results, key=lambda c: c.get("score", 0), reverse=True)

    def hardware_aware(self) -> Dict[str, Any]:
        feasible = [c for c in self.population if self._fits_hardware(c)]
        return feasible[0] if feasible else self.best()

    def _fits_hardware(self, candidate: Dict[str, Any]) -> bool:
        params = candidate["d_model"] * candidate["num_layers"] * candidate["d_ff"] * 2
        mem = candidate["d_model"] * candidate["num_layers"] * 4
        return params / 1e9 < (self.hardware.gpu_memory_gb / 10) and mem < self.hardware.gpu_memory_gb * 1e9

    def recommend(self, task: str = "language_modeling") -> List[Dict[str, Any]]:
        if task == "language_modeling":
            return [c for c in self.population if c["family"] in ("Transformer", "MoE", "NextGen")]
        elif task == "classification":
            return [c for c in self.population if c["d_model"] <= 768]
        else:
            return self.population
