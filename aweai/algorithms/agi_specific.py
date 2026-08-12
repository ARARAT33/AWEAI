from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Concept:
    name: str
    embedding: np.ndarray
    activation: float = 0.0
    salience: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Schema:
    name: str
    slots: Dict[str, Any]
    activation_conditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    actions: List[Callable[[], Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AGIEngine:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(42)
        self._concepts: Dict[str, Concept] = {}
        self._schemas: Dict[str, Schema] = {}
        self._patterns: List[Dict[str, Any]] = []
        self._goals: List[Dict[str, Any]] = []
        self._attention: np.ndarray = np.ones(dim)
        self._cognitive_state: Dict[str, Any] = {"load": 0.0, "focus": 0.0, "fatigue": 0.0}

    def learn_concept(self, name: str, examples: Sequence[np.ndarray]) -> Concept:
        if examples:
            embedding = np.mean(np.array(examples), axis=0)
        else:
            embedding = self._rng.standard_normal(self.dim)
        concept = Concept(name=name, embedding=embedding)
        self._concepts[name] = concept
        return concept

    def activate_concept(self, name: str, input_embedding: np.ndarray, threshold: float = 0.5) -> Optional[Concept]:
        if name not in self._concepts:
            return None
        concept = self._concepts[name]
        similarity = float(np.dot(concept.embedding, input_embedding) / (np.linalg.norm(concept.embedding) * np.linalg.norm(input_embedding) + 1e-8))
        if similarity > threshold:
            concept.activation = similarity
            concept.salience = max(concept.salience, similarity)
            return concept
        return None

    def form_schema(self, name: str, slots: Dict[str, Any], conditions: Optional[List[Callable[[Dict[str, Any]], bool]]] = None, actions: Optional[List[Callable[[], Any]]] = None) -> Schema:
        schema = Schema(name=name, slots=slots, activation_conditions=conditions or [], actions=actions or [])
        self._schemas[name] = schema
        return schema

    def activate_schema(self, context: Dict[str, Any]) -> Optional[Schema]:
        for schema in self._schemas.values():
            if all(cond(context) for cond in schema.activation_conditions):
                for action in schema.actions:
                    action()
                return schema
        return None

    def pattern_complete(self, partial: Dict[str, Any]) -> Dict[str, Any]:
        completed = dict(partial)
        for pattern in self._patterns:
            if all(k in completed for k in pattern.get("keys", [])):
                completed.update(pattern.get("completion", {}))
        return completed

    def pattern_predict(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions = []
        for pattern in self._patterns:
            if all(k in context for k in pattern.get("keys", [])):
                predictions.append(pattern)
        return predictions

    def set_goal(self, description: str, priority: float = 1.0, constraints: Optional[List[str]] = None) -> Dict[str, Any]:
        goal = {"description": description, "priority": priority, "constraints": constraints or [], "status": "active", "progress": 0.0}
        self._goals.append(goal)
        return goal

    def plan(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan = []
        for schema in self._schemas.values():
            if all(c in goal.get("constraints", []) or not goal.get("constraints") for c in schema.slots.get("constraints", [])):
                plan.append({"schema": schema.name, "action": schema.actions[0].__name__ if schema.actions else "none"})
        return plan

    def execute(self, plan: List[Dict[str, Any]]) -> List[Any]:
        results = []
        for step in plan:
            schema = self._schemas.get(step["schema"])
            if schema and schema.actions:
                results.append(schema.actions[0]())
        return results

    def allocate_attention(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        scores = [float(np.dot(self._attention, inp.flatten())) for inp in inputs]
        weights = np.exp(scores) / np.sum(np.exp(scores))
        return weights

    def self_assess(self) -> Dict[str, Any]:
        return {
            "concepts": len(self._concepts),
            "schemas": len(self._schemas),
            "patterns": len(self._patterns),
            "active_goals": sum(1 for g in self._goals if g["status"] == "active"),
            "cognitive_load": self._cognitive_state["load"],
            "attention_focus": self._cognitive_state["focus"],
        }

    def update_cognitive_state(self, load_delta: float = 0.0, focus_delta: float = 0.0, fatigue_delta: float = 0.0) -> None:
        self._cognitive_state["load"] = max(0.0, min(1.0, self._cognitive_state["load"] + load_delta))
        self._cognitive_state["focus"] = max(0.0, min(1.0, self._cognitive_state["focus"] + focus_delta))
        self._cognitive_state["fatigue"] = max(0.0, min(1.0, self._cognitive_state["fatigue"] + fatigue_delta))

    def concepts(self) -> List[str]:
        return list(self._concepts.keys())

    def schemas(self) -> List[str]:
        return list(self._schemas.keys())

    def goals(self) -> List[Dict[str, Any]]:
        return list(self._goals)

    def cognitive_state(self) -> Dict[str, Any]:
        return dict(self._cognitive_state)
