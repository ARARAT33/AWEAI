from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class ReasoningNode:
    id: str
    content: str
    confidence: float
    children: List[ReasoningNode] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChainOfThought:
    def __init__(self) -> None:
        self._chains: List[List[ReasoningNode]] = []

    def reason(self, premises: Sequence[str], conclusion: str) -> List[ReasoningNode]:
        chain: List[ReasoningNode] = []
        for i, premise in enumerate(premises):
            node = ReasoningNode(
                id=f"step_{i}",
                content=premise,
                confidence=0.9,
                metadata={"step": i, "type": "premise"},
            )
            chain.append(node)
        final = ReasoningNode(
            id=f"step_{len(premises)}",
            content=conclusion,
            confidence=0.85,
            metadata={"step": len(premises), "type": "conclusion"},
        )
        if chain:
            final.parent = chain[-1].id
            chain[-1].children.append(final)
        chain.append(final)
        self._chains.append(chain)
        return chain

    def get_chain(self, index: int = -1) -> List[ReasoningNode]:
        return self._chains[index]

    def chains(self) -> List[List[ReasoningNode]]:
        return [list(c) for c in self._chains]


class TreeOfThought:
    def __init__(self, max_depth: int = 5, branching_factor: int = 3) -> None:
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self._trees: List[ReasoningNode] = []

    def explore(self, root_content: str, expand_fn, evaluate_fn) -> Optional[ReasoningNode]:
        root = ReasoningNode(id="root", content=root_content, confidence=evaluate_fn(root_content))

        def _expand(node: ReasoningNode, depth: int) -> None:
            if depth >= self.max_depth:
                return
            branches = expand_fn(node.content)
            branches = branches[: self.branching_factor]
            for branch in branches:
                child = ReasoningNode(
                    id=f"{node.id}_{len(node.children)}",
                    content=branch,
                    confidence=evaluate_fn(branch),
                    parent=node.id,
                )
                node.children.append(child)
                _expand(child, depth + 1)

        _expand(root, 0)
        best = self._best_leaf(root)
        self._trees.append(root)
        return best

    def _best_leaf(self, node: ReasoningNode) -> ReasoningNode:
        if not node.children:
            return node
        best_child = max(node.children, key=lambda c: c.confidence)
        return self._best_leaf(best_child)

    def get_best_path(self) -> List[ReasoningNode]:
        if not self._trees:
            return []
        root = self._trees[-1]
        best = self._best_leaf(root)
        path: List[ReasoningNode] = []
        current: Optional[ReasoningNode] = best
        while current is not None:
            path.append(current)
            if current.parent:
                current = next((n for n in self._tree_nodes(root) if n.id == current.parent), None)
            else:
                current = None
        return list(reversed(path))

    def _tree_nodes(self, root: ReasoningNode) -> List[ReasoningNode]:
        nodes = [root]
        stack = list(root.children)
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes


class GraphOfThought:
    def __init__(self) -> None:
        self._nodes: Dict[str, ReasoningNode] = {}
        self._edges: List[Tuple[str, str]] = []

    def add_node(self, node: ReasoningNode) -> None:
        self._nodes[node.id] = node
        if node.parent and node.parent in self._nodes:
            self._edges.append((node.parent, node.id))

    def add_edge(self, source: str, target: str) -> None:
        if source in self._nodes and target in self._nodes:
            self._edges.append((source, target))

    def propagate(self, source_id: str, decay: float = 0.9) -> None:
        if source_id not in self._nodes:
            return
        source = self._nodes[source_id]
        visited = set()
        queue = [source_id]
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            current = self._nodes[current_id]
            for child in current.children:
                child.confidence = max(0.0, child.confidence * decay + source.confidence * (1 - decay))
                queue.append(child.id)
            for s, t in self._edges:
                if s == current_id:
                    self._nodes[t].confidence = max(0.0, self._nodes[t].confidence * decay + source.confidence * (1 - decay))
                    queue.append(t)

    def highest_confidence(self, top_k: int = 5) -> List[ReasoningNode]:
        scored = [(node.confidence, node) for node in self._nodes.values()]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: {"id": v.id, "content": v.content, "confidence": v.confidence} for k, v in self._nodes.items()},
            "edges": [{"source": s, "target": t} for s, t in self._edges],
        }


class DeductiveEngine:
    def __init__(self) -> None:
        self._axioms: List[str] = []
        self._rules: List[Tuple[str, str]] = []

    def add_axiom(self, axiom: str) -> None:
        self._axioms.append(axiom)

    def add_rule(self, antecedent: str, consequent: str) -> None:
        self._rules.append((antecedent, consequent))

    def modus_ponens(self, antecedent: str, rule: Tuple[str, str]) -> Optional[str]:
        if antecedent == rule[0]:
            return rule[1]
        return None

    def syllogism(self, major: str, minor: str) -> Optional[str]:
        if major.endswith(minor.split(" is ")[0] if " is " in minor else ""):
            return minor
        return None

    def prove(self, hypothesis: str, evidence: Sequence[str]) -> Dict[str, Any]:
        derived: List[str] = list(evidence)
        changed = True
        steps = 0
        while changed and steps < 100:
            changed = False
            steps += 1
            for rule in self._rules:
                for fact in derived:
                    result = self.modus_ponens(fact, rule)
                    if result and result not in derived:
                        derived.append(result)
                        changed = True
        proven = hypothesis in derived
        return {
            "hypothesis": hypothesis,
            "proven": proven,
            "derived_facts": derived,
            "steps": steps,
            "evidence_used": list(evidence),
        }

    def rules_count(self) -> int:
        return len(self._rules)

    def axioms_count(self) -> int:
        return len(self._axioms)


class InductiveEngine:
    def __init__(self, min_confidence: float = 0.8) -> None:
        self.min_confidence = min_confidence
        self._patterns: List[Dict[str, Any]] = []

    def generalize(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(examples) < 2:
            return {"pattern": None, "confidence": 0.0}
        keys = set()
        for ex in examples:
            keys.update(ex.keys())
        pattern: Dict[str, Any] = {}
        for key in keys:
            values = [ex.get(key) for ex in examples if key in ex]
            if len(set(str(v) for v in values)) == 1:
                pattern[key] = values[0]
        confidence = len(pattern) / max(len(keys), 1)
        record = {"pattern": pattern, "confidence": confidence, "examples": len(examples)}
        self._patterns.append(record)
        return record

    def predict(self, pattern: Dict[str, Any], case: Dict[str, Any]) -> Tuple[Any, float]:
        match = 0
        total = len(pattern)
        for key, value in pattern.items():
            if key in case and case[key] == value:
                match += 1
        confidence = match / max(total, 1)
        return pattern.get("target"), confidence

    def patterns_found(self) -> int:
        return len(self._patterns)


class AbductiveEngine:
    def __init__(self) -> None:
        self._hypotheses: List[Dict[str, Any]] = []

    def best_explanation(self, observation: str, hypotheses: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        scored = []
        for hyp in hypotheses:
            likelihood = hyp.get("likelihood", 0.5)
            simplicity = hyp.get("simplicity", 0.5)
            prior = hyp.get("prior", 0.5)
            score = 0.4 * likelihood + 0.3 * simplicity + 0.3 * prior
            scored.append((score, hyp))
        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][1] if scored else None
        if best:
            self._hypotheses.append(best)
        return best

    def hypotheses_count(self) -> int:
        return len(self._hypotheses)


class CausalReasoner:
    def __init__(self) -> None:
        self._graph: Dict[str, List[str]] = {}

    def add_causal_link(self, cause: str, effect: str) -> None:
        self._graph.setdefault(cause, []).append(effect)

    def causes_of(self, event: str) -> List[str]:
        return [cause for cause, effects in self._graph.items() if event in effects]

    def effects_of(self, cause: str) -> List[str]:
        return list(self._graph.get(cause, []))

    def infer_cause(self, observation: str) -> List[str]:
        return self.causes_of(observation)

    def counterfactual(self, event: str, alternative: str) -> Dict[str, Any]:
        original_effects = self.effects_of(event)
        return {
            "original_event": event,
            "alternative": alternative,
            "original_effects": original_effects,
            "counterfactual_outcome": f"If {alternative} instead of {event}, effects would differ",
        }

    def graph(self) -> Dict[str, List[str]]:
        return dict(self._graph)


class CounterfactualReasoner:
    def __init__(self) -> None:
        self._scenarios: List[Dict[str, Any]] = []

    def imagine(self, fact: str, counterfactual: str) -> Dict[str, Any]:
        scenario = {
            "fact": fact,
            "counterfactual": counterfactual,
            "differences": [f"Change from {fact} to {counterfactual}"],
            "implications": ["Unknown - requires further reasoning"],
        }
        self._scenarios.append(scenario)
        return scenario

    def scenarios_count(self) -> int:
        return len(self._scenarios)


class ProofVerifier:
    def __init__(self) -> None:
        self._verified: List[Dict[str, Any]] = []

    def verify_proof(self, proof_steps: Sequence[str], conclusion: str) -> Dict[str, Any]:
        valid = True
        errors: List[str] = []
        for i, step in enumerate(proof_steps):
            if not step or not isinstance(step, str):
                errors.append(f"Invalid step at index {i}")
                valid = False
        if not proof_steps:
            valid = False
            errors.append("Empty proof")
        result = {
            "conclusion": conclusion,
            "valid": valid,
            "steps": len(proof_steps),
            "errors": errors,
        }
        if valid:
            self._verified.append(result)
        return result

    def verify_formal_logic(self, premises: Sequence[str], inference_rule: str, conclusion: str) -> Dict[str, Any]:
        valid_rules = {"modus_ponens", "modus_tollens", "syllogism", "hypothetical_syllogism"}
        if inference_rule not in valid_rules:
            return {"valid": False, "error": f"Unknown inference rule: {inference_rule}"}
        if not premises:
            return {"valid": False, "error": "No premises provided"}
        return {"valid": True, "rule": inference_rule, "conclusion": conclusion, "premises": list(premises)}

    def verified_count(self) -> int:
        return len(self._verified)
