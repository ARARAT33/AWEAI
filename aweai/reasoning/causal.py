from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CausalNode:
    name: str
    values: List[Any]
    parents: List[str]
    cpt: Dict[Tuple[Any, ...], float]


class CausalGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[Tuple[str, str]] = []

    def add_node(self, name: str, values: List[Any], parents: Optional[List[str]] = None, cpt: Optional[Dict[Tuple[Any, ...], float]] = None) -> None:
        self._nodes[name] = CausalNode(name=name, values=values, parents=parents or [], cpt=cpt or {})

    def add_edge(self, parent: str, child: str) -> None:
        self._edges.append((parent, child))
        if child in self._nodes and parent not in self._nodes[child].parents:
            self._nodes[child].parents.append(parent)

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._edges)

    def parents(self, node: str) -> List[str]:
        return list(self._nodes.get(node, CausalNode(name=node, values=[], parents=[])).parents)

    def children(self, node: str) -> List[str]:
        return [child for parent, child in self._edges if parent == node]

    def sample(self, evidence: Optional[Dict[str, Any]] = None, num_samples: int = 100) -> List[Dict[str, Any]]:
        samples = []
        for _ in range(num_samples):
            sample: Dict[str, Any] = {}
            for name, node in self._nodes.items():
                if evidence and name in evidence:
                    sample[name] = evidence[name]
                else:
                    parent_values = tuple(sample.get(p, self._nodes[p].values[0]) for p in node.parents)
                    probs = node.cpt.get(parent_values, [1.0 / len(node.values)] * len(node.values))
                    sample[name] = self._rng.choice(node.values, p=probs)
            samples.append(sample)
        return samples

    def intervene(self, node: str, value: Any, num_samples: int = 100) -> List[Dict[str, Any]]:
        intervention = {node: value}
        return self.sample(evidence=intervention, num_samples=num_samples)

    def counterfactual(self, factual: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
        cf = dict(factual)
        for node, value in intervention.items():
            cf[node] = value
        return cf

    def causal_effect(self, cause: str, effect: str, value: Any, num_samples: int = 100) -> float:
        intervened = self.intervene(cause, value, num_samples)
        return np.mean([s[effect] for s in intervened])

    def backdoor_adjustment(self, treatment: str, outcome: str, adjustment_set: Sequence[str], num_samples: int = 100) -> float:
        controlled = self.sample(num_samples=num_samples)
        values = []
        for s in controlled:
            intervened = self.intervene(treatment, s[treatment], num_samples=1)[0]
            values.append(intervened[outcome])
        return float(np.mean(values))

    def frontdoor_adjustment(self, treatment: str, outcome: str, mediator: str, num_samples: int = 100) -> float:
        total_effect = 0.0
        for _ in range(num_samples):
            s = self.sample(num_samples=1)[0]
            t_value = s[treatment]
            intervened = self.intervene(treatment, t_value, num_samples=1)[0]
            m_value = intervened[mediator]
            cf = self.counterfactual(s, {mediator: m_value})
            total_effect += cf[outcome]
        return total_effect / num_samples

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": {n: {"values": node.values, "parents": node.parents} for n, node in self._nodes.items()}, "edges": self._edges}
