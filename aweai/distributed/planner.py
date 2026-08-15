"""Deterministic task-to-worker planning for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Worker:
    """A worker eligible to receive distributed tasks."""

    name: str
    capacity: int = 1
    healthy: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("worker name must not be empty")
        if self.capacity < 1:
            raise ValueError("worker capacity must be >= 1")


@dataclass(frozen=True)
class Task:
    """A logical task and the number of replicas it requires."""

    name: str
    replicas: int = 1
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must not be empty")
        if self.replicas < 1:
            raise ValueError("task replicas must be >= 1")


class DistributedPlanner:
    """Create deterministic, capacity-aware task placement plans.

    The planner is intentionally side-effect free: it only produces
    ``(task_name, worker_name)`` assignments. Unhealthy workers are ignored.
    Capacity is used as a soft balancing weight, while deterministic tie
    breakers keep repeated plans identical.
    """

    def __init__(self, workers: Iterable[Worker]):
        self.workers = tuple(workers)
        names = [w.name for w in self.workers]
        if len(names) != len(set(names)):
            raise ValueError("worker names must be unique")
        self._healthy = tuple(sorted((w for w in self.workers if w.healthy), key=lambda w: w.name))

    def plan(self, tasks: Iterable[Task]) -> List[Tuple[str, str]]:
        workers = self._healthy
        if not workers:
            raise ValueError("no healthy workers available")

        # Expand replicas in priority order. Higher priority tasks are placed
        # first; task name and replica index provide stable tie breakers.
        expanded = []
        for task in tasks:
            for replica in range(task.replicas):
                expanded.append((task.priority, task.name, replica))
        expanded.sort(key=lambda item: (-item[0], item[1], item[2]))

        load = {w.name: 0 for w in workers}
        result: List[Tuple[str, str]] = []
        for _, task_name, _ in expanded:
            # Choose the worker with the lowest normalized load. Capacity
            # therefore affects balancing without making the output random.
            worker = min(
                workers,
                key=lambda w: (load[w.name] / w.capacity, load[w.name], w.name),
            )
            result.append((task_name, worker.name))
            load[worker.name] += 1
        return result
