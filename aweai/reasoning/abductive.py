from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Hypothesis:
    description: str
    plausibility: float
    prior: float
    likelihood: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbductiveReasoner:
    def __init__(self) -> None:
        self._hypotheses: List[Hypothesis] = []
        self._evidence: List[str] = []
        self._knowledge: Dict[str, float] = {}

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        self._hypotheses.append(hypothesis)

    def add_evidence(self, evidence: str) -> None:
        self._evidence.append(evidence)

    def likelihood(self, hypothesis: str, evidence: str) -> float:
        if hypothesis not in self._knowledge:
            return 0.5
        return self._knowledge[hypothesis]

    def best_explanation(self, observation: str, hypotheses: Optional[List[Hypothesis]] = None) -> Optional[Hypothesis]:
        candidates = hypotheses or self._hypotheses
        if not candidates:
            return None
        scored = []
        for hyp in candidates:
            prior = hyp.prior
            likelihood = self.likelihood(hyp.description, observation)
            plausibility = prior * likelihood
            scored.append((plausibility, hyp))
        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][1]
        best.evidence.append(observation)
        return best

    def explain(self, observation: str) -> List[Dict[str, Any]]:
        explanations = []
        for hyp in self._hypotheses:
            prior = hyp.prior
            likelihood = self.likelihood(hyp.description, observation)
            plausibility = prior * likelihood
            explanations.append({"hypothesis": hyp.description, "plausibility": plausibility, "prior": prior, "likelihood": likelihood})
        explanations.sort(key=lambda x: x["plausibility"], reverse=True)
        return explanations

    def verify(self, hypothesis: Hypothesis, tests: Sequence[Callable[[str], bool]]) -> Dict[str, Any]:
        results = []
        for test in tests:
            try:
                result = test(hypothesis.description)
                results.append(result)
            except Exception:
                results.append(False)
        confidence = sum(results) / max(len(results), 1)
        return {"hypothesis": hypothesis.description, "confidence": confidence, "tests_passed": sum(results), "tests_total": len(results)}

    def abduce(self, observation: str, background_knowledge: Sequence[str]) -> Optional[Hypothesis]:
        for hyp in self._hypotheses:
            if self._is_consistent(hyp, observation, background_knowledge):
                return hyp
        return None

    def _is_consistent(self, hypothesis: Hypothesis, observation: str, background: Sequence[str]) -> bool:
        return hypothesis.description.lower() in observation.lower() or any(hypothesis.description.lower() in b.lower() for b in background)

    def set_knowledge(self, hypothesis: str, probability: float) -> None:
        self._knowledge[hypothesis] = max(0.0, min(1.0, probability))

    def hypotheses(self) -> List[Hypothesis]:
        return list(self._hypotheses)

    def evidence_list(self) -> List[str]:
        return list(self._evidence)
