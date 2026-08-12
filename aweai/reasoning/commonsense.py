from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Frame:
    name: str
    slots: Dict[str, Any]
    defaults: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)


@dataclass
class Script:
    name: str
    steps: List[Dict[str, Any]]
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    actors: List[str] = field(default_factory=list)


class CommonsenseEngine:
    def __init__(self) -> None:
        self._frames: Dict[str, Frame] = {}
        self._scripts: Dict[str, Script] = {}
        self._facts: List[Dict[str, Any]] = []
        self._defaults: Dict[str, Any] = {}

    def add_frame(self, frame: Frame) -> None:
        self._frames[frame.name] = frame

    def add_script(self, script: Script) -> None:
        self._scripts[script.name] = script

    def add_fact(self, fact: Dict[str, Any]) -> None:
        self._facts.append(fact)

    def add_default(self, key: str, value: Any) -> None:
        self._defaults[key] = value

    def apply_frame(self, frame_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        frame = self._frames.get(frame_name)
        if not frame:
            return context
        result = dict(context)
        for slot, default in frame.defaults.items():
            if slot not in result:
                result[slot] = default
        for condition in frame.conditions:
            if not condition(result):
                return {}
        return result

    def execute_script(self, script_name: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        script = self._scripts.get(script_name)
        if not script:
            return []
        results = []
        current = dict(context)
        for step in script.steps:
            if all(p in current for p in step.get("requires", [])):
                result = dict(step)
                result["executed"] = True
                results.append(result)
                current.update(step.get("effects", {}))
            else:
                result = dict(step)
                result["executed"] = False
                results.append(result)
        return results

    def default_reasoning(self, query: str) -> Optional[Any]:
        return self._defaults.get(query)

    def non_monotonic_infer(self, fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for f in reversed(self._facts):
            if self._matches(f, fact):
                return f
        return None

    def _matches(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        for key in fact1:
            if key in fact2 and fact1[key] != fact2[key]:
                return False
        return True

    def context_sensitive_infer(self, context: Dict[str, Any], rules: Sequence[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
        results = []
        for rule in rules:
            if rule(context):
                results.append({"rule": rule.__name__, "fired": True, "context": context})
        return results

    def frame_based_interpret(self, event: Dict[str, Any]) -> Optional[Frame]:
        best_frame = None
        best_score = -1.0
        for frame in self._frames.values():
            score = self._frame_match_score(frame, event)
            if score > best_score:
                best_score = score
                best_frame = frame
        return best_frame

    def _frame_match_score(self, frame: Frame, event: Dict[str, Any]) -> float:
        matches = sum(1 for slot in frame.slots if slot in event and event[slot] == frame.slots[slot])
        return matches / max(len(frame.slots), 1)

    def script_based_predict(self, script_name: str, current_step: int) -> List[Dict[str, Any]]:
        script = self._scripts.get(script_name)
        if not script:
            return []
        return script.steps[current_step:]

    def add_knowledge(self, category: str, knowledge: Dict[str, Any]) -> None:
        self._facts.append({"category": category, **knowledge})

    def query_knowledge(self, category: str, key: str) -> Optional[Any]:
        for fact in self._facts:
            if fact.get("category") == category and key in fact:
                return fact[key]
        return self._defaults.get(key)
