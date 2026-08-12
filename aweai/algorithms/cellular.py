from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Cell:
    x: int
    y: int
    state: int
    neighbors: List["Cell"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reaction:
    reactants: List[str]
    products: List[str]
    rate: float
    catalyst: Optional[str] = None


class CellularAutomaton:
    def __init__(self, width: int, height: int, rule: int = 30, boundary: str = "periodic") -> None:
        self.width = width
        self.height = height
        self.rule = rule
        self.boundary = boundary
        self._rng = np.random.default_rng(42)
        self.grid = self._rng.integers(0, 2, size=(height, width), dtype=int)
        self._history: List[np.ndarray] = [self.grid.copy()]
        self._generation = 0

    def _get_neighborhood(self, x: int, y: int) -> np.ndarray:
        if self.boundary == "periodic":
            neighbors = self.grid.take(range(y - 1, y + 2), mode="wrap", axis=0).take(range(x - 1, x + 2), mode="wrap", axis=1)
        else:
            neighbors = np.pad(self.grid, 1, mode="constant")[y : y + 3, x : x + 3]
        return neighbors

    def _rule_transition(self, neighborhood: np.ndarray) -> int:
        center = neighborhood[1, 1]
        neighbors = [neighborhood[0, 1], neighborhood[1, 0], neighborhood[1, 2], neighborhood[2, 1]]
        idx = sum(2**i if neighbors[i] == 1 else 0 for i in range(4))
        idx += center
        return int((self.rule >> idx) & 1)

    def step(self) -> np.ndarray:
        new_grid = np.zeros_like(self.grid)
        for y in range(self.height):
            for x in range(self.width):
                neighborhood = self._get_neighborhood(x, y)
                new_grid[y, x] = self._rule_transition(neighborhood)
        self.grid = new_grid
        self._history.append(self.grid.copy())
        self._generation += 1
        return self.grid

    def run(self, generations: int) -> List[np.ndarray]:
        for _ in range(generations):
            self.step()
        return list(self._history)

    def entropy(self) -> float:
        values, counts = np.unique(self.grid, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def complexity(self) -> float:
        transitions = 0
        for y in range(self.height):
            for x in range(self.width):
                current = self.grid[y, x]
                if x + 1 < self.width and self.grid[y, x + 1] != current:
                    transitions += 1
                if y + 1 < self.height and self.grid[y + 1, x] != current:
                    transitions += 1
        return transitions / (2 * self.width * self.height)


class LangtonAnt:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self._rng = np.random.default_rng(42)
        self.ant_x = width // 2
        self.ant_y = height // 2
        self.ant_dir = 0
        self._steps = 0
        self._history: List[np.ndarray] = []

    def turn(self, clockwise: bool = True) -> None:
        self.ant_dir = (self.ant_dir + (1 if clockwise else -1)) % 4

    def move(self) -> None:
        if self.ant_dir == 0:
            self.ant_y -= 1
        elif self.ant_dir == 1:
            self.ant_x += 1
        elif self.ant_dir == 2:
            self.ant_y += 1
        elif self.ant_dir == 3:
            self.ant_x -= 1
        self.ant_x = self.ant_x % self.width
        self.ant_y = self.ant_y % self.height

    def step(self) -> np.ndarray:
        current_cell = self.grid[self.ant_y, self.ant_x]
        if current_cell == 0:
            self.turn(clockwise=True)
            self.grid[self.ant_y, self.ant_x] = 1
        else:
            self.turn(clockwise=False)
            self.grid[self.ant_y, self.ant_x] = 0
        self.move()
        self._steps += 1
        if self._steps % 10 == 0:
            self._history.append(self.grid.copy())
        return self.grid

    def run(self, steps: int) -> List[np.ndarray]:
        for _ in range(steps):
            self.step()
        self._history.append(self.grid.copy())
        return list(self._history)

    def highway_length(self) -> int:
        highway_start = None
        length = 0
        for i, g in enumerate(self._history):
            if np.sum(g) > 0 and highway_start is None:
                highway_start = i
            if highway_start is not None:
                length = i - highway_start
        return length


class ArtificialChemistry:
    def __init__(self, num_species: int = 20, max_molecules: int = 100) -> None:
        self.num_species = num_species
        self.max_molecules = max_molecules
        self._rng = np.random.default_rng(42)
        self._species = [f"Molecule_{i}" for i in range(num_species)]
        self._reactions: List[Reaction] = []
        self._population: Dict[str, int] = {s: 0 for s in self._species}
        self._history: List[Dict[str, int]] = []
        self._generate_reactions()

    def _generate_reactions(self, num_reactions: int = 50) -> None:
        for _ in range(num_reactions):
            a = self._rng.choice(self._species)
            b = self._rng.choice(self._species)
            c = self._rng.choice(self._species)
            d = self._rng.choice(self._species)
            rate = self._rng.uniform(0.01, 1.0)
            self._reactions.append(Reaction(reactants=[a, b], products=[c, d], rate=rate))

    def add_molecule(self, species: str, quantity: int = 1) -> None:
        if species in self._population:
            self._population[species] += quantity

    def step(self, dt: float = 1.0) -> None:
        deltas: Dict[str, float] = {s: 0.0 for s in self._species}
        for reaction in self._reactions:
            a, b = reaction.reactants
            rate = reaction.rate * self._population.get(a, 0) * self._population.get(b, 0) * dt
            for product in reaction.products:
                deltas[product] += rate
            deltas[a] -= rate
            deltas[b] -= rate
        for species in self._species:
            self._population[species] = max(0, self._population[species] + int(deltas[species]))
        self._history.append(dict(self._population))

    def run(self, steps: int = 100) -> None:
        for _ in range(steps):
            self.step()

    def diversity(self) -> float:
        active = sum(1 for v in self._population.values() if v > 0)
        return active / max(len(self._species), 1)

    def concentration(self, species: str) -> int:
        return self._population.get(species, 0)


class AgentBasedModel:
    def __init__(self, num_agents: int = 100, width: int = 50, height: int = 50) -> None:
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self._rng = np.random.default_rng(42)
        self.agents = [{"id": i, "x": self._rng.integers(0, width), "y": self._rng.integers(0, height), "energy": 1.0, "wealth": 0.0} for i in range(num_agents)]
        self._grid = np.zeros((height, width), dtype=int)
        self._history: List[Dict[str, Any]] = []

    def step(self, interaction_radius: float = 5.0, wealth_transfer_rate: float = 0.1) -> None:
        for agent in self.agents:
            agent["x"] = (agent["x"] + self._rng.integers(-1, 2)) % self.width
            agent["y"] = (agent["y"] + self._rng.integers(-1, 2)) % self.height
            agent["energy"] = max(0, agent["energy"] - 0.01)
            agent["wealth"] += agent["energy"] * 0.1
        self._grid[:, :] = 0
        for agent in self.agents:
            self._grid[agent["y"], agent["x"]] += 1
        self._history.append({"agents": [dict(a) for a in self.agents], "grid": self._grid.tolist()})

    def run(self, steps: int = 100) -> None:
        for _ in range(steps):
            self.step()

    def wealth_distribution(self) -> Dict[str, float]:
        wealths = [a["wealth"] for a in self.agents]
        return {"mean": float(np.mean(wealths)), "std": float(np.std(wealths)), "min": float(np.min(wealths)), "max": float(np.max(wealths)), "gini": self._gini(wealths)}

    def _gini(self, values: Sequence[float]) -> float:
        values = np.array(sorted(values))
        n = len(values)
        if n == 0:
            return 0.0
        cumsum = np.cumsum(values)
        return float((2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values)))
