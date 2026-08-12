from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Idea:
    content: str
    novelty: float
    value: float
    feasibility: float
    parent_ideas: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CreativeEngine:
    def __init__(self, novelty_weight: float = 0.4, value_weight: float = 0.4, feasibility_weight: float = 0.2) -> None:
        self.novelty_weight = novelty_weight
        self.value_weight = value_weight
        self.feasibility_weight = feasibility_weight
        self._ideas: List[Idea] = []
        self._concepts: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(42)

    def register_concept(self, name: str, embedding: np.ndarray) -> None:
        self._concepts[name] = np.asarray(embedding, dtype=float)

    def _compute_novelty(self, idea_embedding: np.ndarray) -> float:
        if not self._concepts:
            return 1.0
        similarities = []
        for concept_emb in self._concepts.values():
            sim = float(np.dot(idea_embedding, concept_emb) / (np.linalg.norm(idea_embedding) * np.linalg.norm(concept_emb) + 1e-8))
            similarities.append(sim)
        max_sim = max(similarities) if similarities else 0.0
        return 1.0 - max_sim

    def _compute_value(self, idea_embedding: np.ndarray, criteria: Sequence[np.ndarray]) -> float:
        if not criteria:
            return 0.5
        scores = [float(np.dot(idea_embedding, c) / (np.linalg.norm(idea_embedding) * np.linalg.norm(c) + 1e-8)) for c in criteria]
        return float(np.mean(scores))

    def _compute_feasibility(self, idea_embedding: np.ndarray, constraints: Sequence[np.ndarray]) -> float:
        if not constraints:
            return 1.0
        violations = 0
        for constraint in constraints:
            if float(np.dot(idea_embedding, constraint)) < 0:
                violations += 1
        return 1.0 - violations / max(len(constraints), 1)

    def generate_ideas(self, num_ideas: int, concept_names: Sequence[str], criteria: Optional[Sequence[np.ndarray]] = None, constraints: Optional[Sequence[np.ndarray]] = None) -> List[Idea]:
        ideas = []
        concept_embeddings = [self._concepts[name] for name in concept_names if name in self._concepts]
        for _ in range(num_ideas):
            if concept_embeddings:
                embedding = sum(self._rng.normal(0, 0.1, len(c)) + c for c in self._rng.choice(concept_embeddings, size=min(2, len(concept_embeddings)), replace=False))
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            else:
                embedding = self._rng.standard_normal(50)
            novelty = self._compute_novelty(embedding)
            value = self._compute_value(embedding, criteria or [])
            feasibility = self._compute_feasibility(embedding, constraints or [])
            score = self.novelty_weight * novelty + self.value_weight * value + self.feasibility_weight * feasibility
            idea = Idea(content=f"idea_{len(self._ideas)}", novelty=novelty, value=value, feasibility=feasibility, metadata={"score": score, "embedding": embedding.tolist()})
            ideas.append(idea)
            self._ideas.append(idea)
        return ideas

    def combine_ideas(self, idea1: Idea, idea2: Idea, blend_ratio: float = 0.5) -> Idea:
        emb1 = np.array(idea1.metadata.get("embedding", self._rng.standard_normal(50)))
        emb2 = np.array(idea2.metadata.get("embedding", self._rng.standard_normal(50)))
        combined_emb = blend_ratio * emb1 + (1 - blend_ratio) * emb2
        novelty = self._compute_novelty(combined_emb)
        value = (idea1.value + idea2.value) / 2
        feasibility = min(idea1.feasibility, idea2.feasibility)
        return Idea(content=f"blend_{idea1.content}_{idea2.content}", novelty=novelty, value=value, feasibility=feasibility, parent_ideas=[idea1.content, idea2.content], metadata={"score": self.novelty_weight * novelty + self.value_weight * value + self.feasibility_weight * feasibility, "embedding": combined_emb.tolist()})

    def divergent_thinking(self, seed_idea: Idea, num_variations: int = 10, variation_scale: float = 0.3) -> List[Idea]:
        variations = []
        seed_emb = np.array(seed_idea.metadata.get("embedding", self._rng.standard_normal(50)))
        for i in range(num_variations):
            variation_emb = seed_emb + self._rng.normal(0, variation_scale, len(seed_emb))
            variation_emb = variation_emb / (np.linalg.norm(variation_emb) + 1e-8)
            novelty = self._compute_novelty(variation_emb)
            value = seed_idea.value * (0.8 + 0.4 * self._rng.random())
            feasibility = seed_idea.feasibility * (0.7 + 0.6 * self._rng.random())
            variation = Idea(content=f"variation_{seed_idea.content}_{i}", novelty=novelty, value=value, feasibility=feasibility, parent_ideas=[seed_idea.content], metadata={"score": self.novelty_weight * novelty + self.value_weight * value + self.feasibility_weight * feasibility, "embedding": variation_emb.tolist()})
            variations.append(variation)
            self._ideas.append(variation)
        return variations

    def convergent_thinking(self, ideas: Sequence[Idea], top_k: int = 3) -> List[Idea]:
        scored = [(idea.metadata.get("score", 0.0), idea) for idea in ideas]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [idea for _, idea in scored[:top_k]]

    def evaluate_aesthetics(self, idea: Idea, aesthetic_criteria: Sequence[np.ndarray]) -> float:
        emb = np.array(idea.metadata.get("embedding", self._rng.standard_normal(50)))
        if not aesthetic_criteria:
            return 0.5
        scores = [float(np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-8)) for c in aesthetic_criteria]
        return float(np.mean(scores))

    def detect_novelty(self, idea: Idea, threshold: float = 0.7) -> bool:
        return idea.novelty > threshold

    def idea_history(self) -> List[Idea]:
        return list(self._ideas)

    def best_ideas(self, top_k: int = 10) -> List[Idea]:
        scored = [(idea.metadata.get("score", 0.0), idea) for idea in self._ideas]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [idea for _, idea in scored[:top_k]]


class ImprovisationEngine:
    def __init__(self, repertoire_size: int = 100) -> None:
        self.repertoire_size = repertoire_size
        self._repertoire: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(42)

    def learn(self, patterns: Sequence[Dict[str, Any]]) -> None:
        self._repertoire = list(patterns)[: self.repertoire_size]

    def improvise(self, context: Dict[str, Any], length: int = 10) -> List[Dict[str, Any]]:
        if not self._repertoire:
            return []
        result = []
        current = dict(context)
        for _ in range(length):
            template = self._rng.choice(self._repertoire)
            variation = dict(template)
            variation.update(current)
            for key in variation:
                if isinstance(variation[key], (int, float)):
                    variation[key] += self._rng.normal(0, 0.1)
            result.append(variation)
        return result

    def evaluate_serendipity(self, discovered: Dict[str, Any], expected: Dict[str, Any]) -> float:
        if not expected:
            return 1.0
        matches = sum(1 for k, v in expected.items() if k in discovered and discovered[k] == v)
        return matches / max(len(expected), 1)
