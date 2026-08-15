"""Portable checkpoint/recovery primitives for AWEAI workloads."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Mapping
import json


@dataclass(frozen=True)
class Checkpoint:
    workload: str
    step: int
    state_digest: str
    metadata_digest: str

    @property
    def key(self) -> str:
        return f"{self.workload}:{self.step}:{self.state_digest[:16]}"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


class CheckpointStore:
    """In-memory reference store; adapters can persist the same contract remotely."""

    def __init__(self) -> None:
        self._items: Dict[str, Checkpoint] = {}

    def save(self, workload: str, step: int, state: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> Checkpoint:
        if step < 0:
            raise ValueError("step must be non-negative")
        checkpoint = Checkpoint(workload, step, _digest(state), _digest(metadata or {}))
        self._items[checkpoint.key] = checkpoint
        return checkpoint

    def latest(self, workload: str) -> Checkpoint | None:
        matches = [c for c in self._items.values() if c.workload == workload]
        return max(matches, key=lambda c: c.step) if matches else None

    def verify(self, checkpoint: Checkpoint, state: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> bool:
        return checkpoint.state_digest == _digest(state) and checkpoint.metadata_digest == _digest(metadata or {})
