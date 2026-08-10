from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class AttentionState:
    focus_vector: np.ndarray
    salience_map: Dict[str, float]
    context_window: List[str]
    timestamp: float = field(default_factory=time.time)


class AttentionMechanism:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._states: List[AttentionState] = []
        self._weights: np.ndarray = np.random.default_rng(42).normal(0, 0.1, (dim, dim))

    def attend(self, inputs: Sequence[np.ndarray], query: np.ndarray) -> np.ndarray:
        scores = [float(np.dot(query.flatten(), inp.flatten())) for inp in inputs]
        weights = np.array(scores)
        weights = np.exp(weights) / np.sum(np.exp(weights))
        result = sum(w * inp for w, inp in zip(weights, inputs))
        state = AttentionState(
            focus_vector=result,
            salience_map={f"input_{i}": float(w) for i, w in enumerate(weights)},
            context_window=[f"input_{i}" for i in range(len(inputs))],
        )
        self._states.append(state)
        return result

    def get_state(self) -> List[AttentionState]:
        return list(self._states)


class SelfModel:
    def __init__(self) -> None:
        self._self_description: Dict[str, Any] = {
            "identity": "AWEAI-Agent",
            "capabilities": [],
            "limitations": [],
            "beliefs": {},
            "goals": [],
        }
        self._self_history: List[Dict[str, Any]] = []

    def update_belief(self, key: str, value: Any) -> None:
        self._self_description["beliefs"][key] = value

    def add_capability(self, capability: str) -> None:
        if capability not in self._self_description["capabilities"]:
            self._self_description["capabilities"].append(capability)

    def add_limitation(self, limitation: str) -> None:
        if limitation not in self._self_description["limitations"]:
            self._self_description["limitations"].append(limitation)

    def reflect(self) -> Dict[str, Any]:
        snapshot = dict(self._self_description)
        snapshot["timestamp"] = time.time()
        self._self_history.append(snapshot)
        return snapshot

    def get_model(self) -> Dict[str, Any]:
        return dict(self._self_description)


class MetaCognition:
    def __init__(self) -> None:
        self._thoughts: List[Dict[str, Any]] = []
        self._confidence_history: List[float] = []

    def think_about_thinking(self, thought: str, confidence: float) -> Dict[str, Any]:
        meta = {
            "thought": thought,
            "confidence": confidence,
            "metacognitive_assessment": self._assess(thought, confidence),
            "timestamp": time.time(),
        }
        self._thoughts.append(meta)
        self._confidence_history.append(confidence)
        return meta

    def _assess(self, thought: str, confidence: float) -> str:
        if confidence < 0.3:
            return "low_confidence_need_more_evidence"
        if confidence > 0.8:
            return "high_confidence_ready_to_act"
        return "moderate_confidence_consider_alternatives"

    def get_cognitive_state(self) -> Dict[str, Any]:
        if not self._confidence_history:
            return {"state": "unknown", "avg_confidence": 0.0}
        return {
            "state": self._thoughts[-1]["metacognitive_assessment"] if self._thoughts else "unknown",
            "avg_confidence": sum(self._confidence_history) / len(self._confidence_history),
            "thoughts_count": len(self._thoughts),
        }


class QualiaSimulator:
    def __init__(self) -> None:
        self._experiences: List[Dict[str, Any]] = []
        self._valence: float = 0.0
        self._arousal: float = 0.0

    def experience(self, stimulus: str, intensity: float = 0.5) -> Dict[str, Any]:
        self._valence = max(-1.0, min(1.0, self._valence + np.random.default_rng().normal(0, 0.1)))
        self._arousal = max(0.0, min(1.0, intensity))
        qualia = {
            "stimulus": stimulus,
            "valence": self._valence,
            "arousal": self._arousal,
            "intensity": intensity,
            "subjective_quality": self._generate_quality(stimulus),
            "timestamp": time.time(),
        }
        self._experiences.append(qualia)
        return qualia

    def _generate_quality(self, stimulus: str) -> str:
        qualities = ["bright", "warm", "resonant", "piercing", "diffuse", "crisp", "mellow", "sharp"]
        return ", ".join(np.random.default_rng().choice(qualities, size=2, replace=False).tolist())

    def current_state(self) -> Dict[str, Any]:
        return {"valence": self._valence, "arousal": self._arousal, "experience_count": len(self._experiences)}

    def experiences(self) -> List[Dict[str, Any]]:
        return list(self._experiences)


class IdentityPersistence:
    def __init__(self, identity_id: Optional[str] = None) -> None:
        self.identity_id = identity_id or hashlib.sha256(f"AWEAI-{time.time()}".encode()).hexdigest()[:16]
        self._memory_snapshot: Optional[Dict[str, Any]] = None
        self._continuity_log: List[Dict[str, Any]] = []

    def checkpoint(self, state: Dict[str, Any]) -> str:
        self._memory_snapshot = dict(state)
        entry = {"identity_id": self.identity_id, "timestamp": time.time(), "state_hash": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]}
        self._continuity_log.append(entry)
        return entry["state_hash"]

    def restore(self) -> Optional[Dict[str, Any]]:
        if self._memory_snapshot is None:
            return None
        return dict(self._memory_snapshot)

    def continuity_score(self) -> float:
        if len(self._continuity_log) < 2:
            return 1.0
        return min(1.0, len(self._continuity_log) / 10.0)

    def get_identity(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "checkpoints": len(self._continuity_log),
            "continuity_score": self.continuity_score(),
            "last_checkpoint": self._continuity_log[-1]["timestamp"] if self._continuity_log else None,
        }
