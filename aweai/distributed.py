"""Framework-neutral distributed execution primitives for AWEAI."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Worker:
    name: str
    capacity: int = 1
    healthy: bool = True


@dataclass(frozen=True)
class Task:
    name: str
    replicas: int = 1
    priority: int = 0


class DistributedPlanner:
    """Deterministically assign replicas across healthy workers."""

    def __init__(self, workers: Iterable[Worker]):
        self.workers = tuple(workers)
        if not self.workers:
            raise ValueError("at least one worker is required")

    def plan(self, tasks: Iterable[Task]) -> List[Tuple[str, str]]:
        workers = [w for w in self.workers if w.healthy and w.capacity > 0]
        if not workers:
            raise RuntimeError("no healthy worker capacity")
        capacity = {w.name: w.capacity for w in workers}
        assignments: List[Tuple[str, str]] = []
        for task in sorted(tasks, key=lambda x: (-x.priority, x.name)):
            if task.replicas < 1:
                raise ValueError(f"{task.name}: replicas must be positive")
            for _ in range(task.replicas):
                candidates = [name for name, slots in capacity.items() if slots > 0]
                if not candidates:
                    raise RuntimeError(f"insufficient capacity for {task.name}")
                worker = min(candidates, key=lambda name: (-capacity[name], name))
                assignments.append((task.name, worker))
                capacity[worker] -= 1
        return assignments

    def fingerprint(self, tasks: Iterable[Task]) -> str:
        payload = "|".join(f"{task}:{worker}" for task, worker in self.plan(tasks))
        return sha256(payload.encode()).hexdigest()
