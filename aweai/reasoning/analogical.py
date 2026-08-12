from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Concept:
    name: str
    embedding: np.ndarray
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Analogy:
    source: Concept
    target: Concept
    mapping: Dict[str, str]
    similarity: float
    validity: float


class AnalogyEngine:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(42)
        self._concepts: Dict[str, Concept] = {}

    def add_concept(self, name: str, embedding: Optional[np.ndarray] = None, attributes: Optional[Dict[str, Any]] = None) -> Concept:
        if embedding is None:
            embedding = self._rng.standard_normal(self.dim)
        concept = Concept(name=name, embedding=np.asarray(embedding, dtype=float), attributes=attributes or {})
        self._concepts[name] = concept
        return concept

    def similarity(self, concept_a: str, concept_b: str) -> float:
        if concept_a not in self._concepts or concept_b not in self._concepts:
            return 0.0
        a = self._concepts[concept_a].embedding
        b = self._concepts[concept_b].embedding
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def find_analogy(self, source_a: str, source_b: str, targets: Sequence[str]) -> Optional[Analogy]:
        if source_a not in self._concepts or source_b not in self._concepts:
            return None
        a = self._concepts[source_a].embedding
        b = self._concepts[source_b].embedding
        transformation = b - a
        best_target = None
        best_similarity = -1.0
        for target_name in targets:
            if target_name not in self._concepts:
                continue
            t = self._concepts[target_name].embedding
            predicted = t + transformation
            for candidate_name in self._concepts:
                if candidate_name == target_name:
                    continue
                sim = float(np.dot(predicted, self._concepts[candidate_name].embedding) / (np.linalg.norm(predicted) * np.linalg.norm(self._concepts[candidate_name].embedding) + 1e-8))
                if sim > best_similarity:
                    best_similarity = sim
                    best_target = candidate_name
        if best_target is None:
            return None
        mapping = {source_a: source_b, source_a: best_target}
        return Analogy(source=self._concepts[source_a], target=self._concepts[best_target], mapping=mapping, similarity=best_similarity, validity=best_similarity)

    def transfer_knowledge(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        analogies = []
        for s_name, s_concept in self._concepts.items():
            if not s_concept.attributes.get("domain") == source_domain:
                continue
            for t_name, t_concept in self._concepts.items():
                if t_concept.attributes.get("domain") != target_domain:
                    continue
                sim = self.similarity(s_name, t_name)
                if sim > 0.5:
                    analogies.append({"source": s_name, "target": t_name, "similarity": sim})
        return {"domain_transfer": source_domain + " -> " + target_domain, "analogies": analogies, "count": len(analogies)}

    def case_based_reason(self, problem: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best_case = None
        best_score = -1.0
        for case in cases:
            score = self._case_similarity(problem, case)
            if score > best_score:
                best_score = score
                best_case = case
        return best_case

    def _case_similarity(self, problem: Dict[str, Any], case: Dict[str, Any]) -> float:
        common_keys = set(problem.keys()) & set(case.keys())
        if not common_keys:
            return 0.0
        matches = sum(1 for k in common_keys if problem[k] == case[k])
        return matches / max(len(common_keys), 1)

    def metaphor_generate(self, concept_a: str, concept_b: str) -> Dict[str, Any]:
        sim = self.similarity(concept_a, concept_b)
        return {"metaphor": f"{concept_a} is {concept_b}", "similarity": sim, "mapping": {concept_a: concept_b}}

    def concepts(self) -> List[str]:
        return list(self._concepts.keys())
