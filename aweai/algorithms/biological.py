from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class SpikingNeuron:
    id: str
    threshold: float = 1.0
    membrane_potential: float = 0.0
    decay: float = 0.1
    last_spike_time: float = -1.0
    spike_history: List[float] = field(default_factory=list)
    synaptic_weights: Dict[str, float] = field(default_factory=dict)

    def integrate(self, input_current: float, dt: float = 1.0) -> bool:
        self.membrane_potential += input_current * dt
        self.membrane_potential -= self.decay * self.membrane_potential * dt
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.last_spike_time = time.time()
            self.spike_history.append(self.last_spike_time)
            return True
        return False

    def reset(self) -> None:
        self.membrane_potential = 0.0
        self.last_spike_time = -1.0


class SpikingNeuralNetwork:
    def __init__(self, num_neurons: int = 100, connection_probability: float = 0.1) -> None:
        self.num_neurons = num_neurons
        self.connection_probability = connection_probability
        self._rng = np.random.default_rng(42)
        self.neurons: List[SpikingNeuron] = [
            SpikingNeuron(id=f"neuron_{i}", threshold=self._rng.uniform(0.8, 1.2), decay=self._rng.uniform(0.05, 0.2)) for i in range(num_neurons)
        ]
        self._connectivity: Dict[str, List[str]] = {}
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        for i, neuron in enumerate(self.neurons):
            targets = []
            for j in range(self.num_neurons):
                if i != j and self._rng.random() < self.connection_probability:
                    targets.append(f"neuron_{j}")
                    weight = self._rng.normal(0, 0.5)
                    self.neurons[j].synaptic_weights[f"neuron_{i}"] = weight
            self._connectivity[neuron.id] = targets

    def step(self, input_currents: Sequence[float], dt: float = 1.0) -> List[str]:
        spikes = []
        for neuron, current in zip(self.neurons, input_currents):
            if neuron.integrate(float(current), dt=dt):
                spikes.append(neuron.id)
        return spikes

    def propagate_spikes(self, spikes: Sequence[str]) -> Dict[str, float]:
        post_synaptic_currents = {neuron.id: 0.0 for neuron in self.neurons}
        for pre_id in spikes:
            pre_neuron = next((n for n in self.neurons if n.id == pre_id), None)
            if pre_neuron is None:
                continue
            for post_id in self._connectivity.get(pre_id, []):
                post_neuron = next((n for n in self.neurons if n.id == post_id), None)
                if post_neuron is None:
                    continue
                weight = post_neuron.synaptic_weights.get(pre_id, 0.0)
                post_synaptic_currents[post_id] += weight
        return post_synaptic_currents

    def run_simulation(self, steps: int = 100, input_fn: Optional[Callable[[int], Sequence[float]]] = None) -> List[Dict[str, Any]]:
        trace = []
        for t in range(steps):
            inputs = input_fn(t) if input_fn is not None else [0.0] * self.num_neurons
            spikes = self.step(inputs)
            currents = self.propagate_spikes(spikes)
            trace.append({"time": t, "spikes": spikes, "currents": dict(currents)})
        return trace


class NeuromorphicPattern:
    def __init__(self, pattern_size: int = 10, num_patterns: int = 5) -> None:
        self.pattern_size = pattern_size
        self.num_patterns = num_patterns
        self._rng = np.random.default_rng(42)
        self._patterns = [self._rng.choice([-1, 1], size=pattern_size).astype(float) for _ in range(num_patterns)]

    def recall(self, noisy_pattern: np.ndarray, steps: int = 10) -> np.ndarray:
        state = noisy_pattern.copy()
        for _ in range(steps):
            for pattern in self._patterns:
                correlation = float(np.dot(state, pattern))
                if correlation > 0:
                    state += 0.1 * pattern
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm
        return state

    def store_pattern(self, pattern: np.ndarray) -> None:
        self._patterns.append(pattern.copy())
        self.num_patterns = len(self._patterns)

    def pattern_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class DNASimulator:
    def __init__(self, sequence_length: int = 100) -> None:
        self.sequence_length = sequence_length
        self._bases = ["A", "T", "G", "C"]
        self._rng = np.random.default_rng(42)
        self.sequence = [self._rng.choice(self._bases) for _ in range(sequence_length)]

    def mutate(self, rate: float = 0.01) -> None:
        for i in range(self.sequence_length):
            if self._rng.random() < rate:
                self.sequence[i] = self._rng.choice(self._bases)

    def crossover(self, other: "DNASimulator", crossover_point: Optional[int] = None) -> "DNASimulator":
        if crossover_point is None:
            crossover_point = self._rng.integers(1, self.sequence_length)
        new_seq = self.sequence[:crossover_point] + other.sequence[crossover_point:]
        child = DNASimulator(sequence_length=self.sequence_length)
        child.sequence = new_seq
        return child

    def fitness(self, target: Sequence[str]) -> float:
        matches = sum(1 for a, b in zip(self.sequence, target) if a == b)
        return matches / max(len(self.sequence), len(target))

    def to_binary(self) -> np.ndarray:
        mapping = {"A": [0, 0], "T": [0, 1], "G": [1, 0], "C": [1, 1]}
        return np.array([mapping[b] for b in self.sequence], dtype=float).flatten()


class ImmuneSystemAlgorithm:
    def __init__(self, population_size: int = 50, clone_rate: int = 5, mutation_rate: float = 0.01) -> None:
        self.population_size = population_size
        self.clone_rate = clone_rate
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(42)
        self._antibodies: List[np.ndarray] = [self._rng.standard_normal(10) for _ in range(population_size)]
        self._memory_cells: List[np.ndarray] = []

    def affinity(self, antibody: np.ndarray, antigen: np.ndarray) -> float:
        return float(np.dot(antibody, antigen) / (np.linalg.norm(antibody) * np.linalg.norm(antigen) + 1e-8))

    def select_clones(self, antigen: np.ndarray) -> List[np.ndarray]:
        affinities = [(ab, self.affinity(ab, antigen)) for ab in self._antibodies]
        affinities.sort(key=lambda x: x[1], reverse=True)
        clones = []
        for ab, _ in affinities[: self.population_size // 2]:
            for _ in range(self.clone_rate):
                clone = ab.copy()
                mask = self._rng.random(ab.shape) < self.mutation_rate
                clone[mask] += self._rng.normal(0, 0.1, size=ab.shape)[mask]
                clones.append(clone)
        return clones

    def update_population(self, antigen: np.ndarray) -> None:
        clones = self.select_clones(antigen)
        scored = [(ab, self.affinity(ab, antigen)) for ab in clones]
        scored.sort(key=lambda x: x[1], reverse=True)
        self._antibodies = [ab for ab, _ in scored[: self.population_size]]
        best = self._antibodies[0]
        if self.affinity(best, antigen) > 0.8:
            self._memory_cells.append(best.copy())

    def detect(self, pattern: np.ndarray, threshold: float = 0.7) -> bool:
        for ab in self._memory_cells:
            if self.affinity(ab, pattern) > threshold:
                return True
        return False


class AntColonyOptimization:
    def __init__(self, num_ants: int = 30, num_cities: int = 20, alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.5) -> None:
        self.num_ants = num_ants
        self.num_cities = num_cities
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self._rng = np.random.default_rng(42)
        self._cities = self._rng.standard_normal((num_cities, 2))
        self._distances = np.linalg.norm(self._cities[:, None] - self._cities[None, :], axis=2)
        self._pheromones = np.ones((num_cities, num_cities))

    def _probability(self, current: int, visited: List[int]) -> np.ndarray:
        unvisited = [i for i in range(self.num_cities) if i not in visited]
        if not unvisited:
            return np.zeros(self.num_cities)
        probs = np.zeros(self.num_cities)
        for j in unvisited:
            tau = self._pheromones[current, j] ** self.alpha
            eta = (1.0 / (self._distances[current, j] + 1e-8)) ** self.beta
            probs[j] = tau * eta
        total = probs.sum()
        if total == 0:
            probs = np.ones(self.num_cities) / self.num_cities
        else:
            probs = probs / total
        return probs

    def construct_solution(self) -> List[int]:
        start = self._rng.integers(0, self.num_cities)
        tour = [start]
        current = start
        while len(tour) < self.num_cities:
            probs = self._probability(current, tour)
            next_city = int(self._rng.choice(self.num_cities, p=probs))
            tour.append(next_city)
            current = next_city
        return tour

    def tour_length(self, tour: Sequence[int]) -> float:
        return sum(self._distances[tour[i], tour[(i + 1) % self.num_cities]] for i in range(len(tour)))

    def step(self) -> None:
        tours = [self.construct_solution() for _ in range(self.num_ants)]
        lengths = [self.tour_length(t) for t in tours]
        self._pheromones *= 1 - self.evaporation
        for tour, length in zip(tours, lengths):
            deposit = 1.0 / (length + 1e-8)
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                self._pheromones[a, b] += deposit
                self._pheromones[b, a] += deposit

    def best_tour(self, iterations: int = 100) -> Tuple[List[int], float]:
        best_tour = self.construct_solution()
        best_length = self.tour_length(best_tour)
        for _ in range(iterations):
            self.step()
            tour = self.construct_solution()
            length = self.tour_length(tour)
            if length < best_length:
                best_length = length
                best_tour = tour
        return best_tour, best_length


class BeeColonyOptimization:
    def __init__(self, num_bees: int = 50, num_food_sources: int = 25, limit: int = 10) -> None:
        self.num_bees = num_bees
        self.num_food_sources = num_food_sources
        self.limit = limit
        self._rng = np.random.default_rng(42)
        self._sources = [self._rng.standard_normal(10) for _ in range(num_food_sources)]
        self._fitness = [0.0] * num_food_sources
        self._trials = [0] * num_food_sources

    def evaluate(self, source: np.ndarray, fitness_fn: Callable[[np.ndarray], float]) -> float:
        return fitness_fn(source)

    def employed_bee_phase(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        for i, source in enumerate(self._sources):
            neighbor_idx = self._rng.integers(0, self.num_food_sources)
            while neighbor_idx == i:
                neighbor_idx = self._rng.integers(0, self.num_food_sources)
            phi = self._rng.uniform(-1, 1)
            new_source = source + phi * (source - self._sources[neighbor_idx])
            new_fitness = self.evaluate(new_source, fitness_fn)
            if new_fitness > self._fitness[i]:
                self._sources[i] = new_source
                self._fitness[i] = new_fitness
                self._trials[i] = 0
            else:
                self._trials[i] += 1

    def onlooker_bee_phase(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        total_fitness = sum(max(0.0, f) for f in self._fitness)
        if total_fitness == 0:
            probs = [1.0 / self.num_food_sources] * self.num_food_sources
        else:
            probs = [max(0.0, f) / total_fitness for f in self._fitness]
        for _ in range(self.num_bees // 2):
            idx = int(self._rng.choice(self.num_food_sources, p=probs))
            source = self._sources[idx]
            neighbor_idx = self._rng.integers(0, self.num_food_sources)
            while neighbor_idx == idx:
                neighbor_idx = self._rng.integers(0, self.num_food_sources)
            phi = self._rng.uniform(-1, 1)
            new_source = source + phi * (source - self._sources[neighbor_idx])
            new_fitness = self.evaluate(new_source, fitness_fn)
            if new_fitness > self._fitness[idx]:
                self._sources[idx] = new_source
                self._fitness[idx] = new_fitness
                self._trials[idx] = 0

    def scout_bee_phase(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        for i in range(self.num_food_sources):
            if self._trials[i] > self.limit:
                self._sources[i] = self._rng.standard_normal(10)
                self._fitness[i] = self.evaluate(self._sources[i], fitness_fn)
                self._trials[i] = 0

    def optimize(self, fitness_fn: Callable[[np.ndarray], float], iterations: int = 100) -> Tuple[np.ndarray, float]:
        self._fitness = [self.evaluate(s, fitness_fn) for s in self._sources]
        best_idx = int(np.argmax(self._fitness))
        best_source = self._sources[best_idx].copy()
        best_fitness = self._fitness[best_idx]
        for _ in range(iterations):
            self.employed_bee_phase(fitness_fn)
            self.onlooker_bee_phase(fitness_fn)
            self.scout_bee_phase(fitness_fn)
            current_best = max(self._fitness)
            if current_best > best_fitness:
                best_fitness = current_best
                best_idx = int(np.argmax(self._fitness))
                best_source = self._sources[best_idx].copy()
        return best_source, best_fitness


class ParticleSwarmOptimization:
    def __init__(self, num_particles: int = 30, dim: int = 10, w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> None:
        self.num_particles = num_particles
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self._rng = np.random.default_rng(42)
        self._positions = self._rng.standard_normal((num_particles, dim))
        self._velocities = self._rng.standard_normal((num_particles, dim)) * 0.1
        self._best_positions = self._positions.copy()
        self._best_fitness = [-math.inf] * num_particles
        self._global_best_position = self._positions[0].copy()
        self._global_best_fitness = -math.inf

    def evaluate(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        for i in range(self.num_particles):
            fitness = fitness_fn(self._positions[i])
            if fitness > self._best_fitness[i]:
                self._best_fitness[i] = fitness
                self._best_positions[i] = self._positions[i].copy()
            if fitness > self._global_best_fitness:
                self._global_best_fitness = fitness
                self._global_best_position = self._positions[i].copy()

    def step(self) -> None:
        for i in range(self.num_particles):
            r1, r2 = self._rng.random(2)
            self._velocities[i] = (
                self.w * self._velocities[i]
                + self.c1 * r1 * (self._best_positions[i] - self._positions[i])
                + self.c2 * r2 * (self._global_best_position - self._positions[i])
            )
            self._velocities[i] = np.clip(self._velocities[i], -1.0, 1.0)
            self._positions[i] += self._velocities[i]

    def optimize(self, fitness_fn: Callable[[np.ndarray], float], iterations: int = 100) -> Tuple[np.ndarray, float]:
        self.evaluate(fitness_fn)
        for _ in range(iterations):
            self.step()
            self.evaluate(fitness_fn)
        return self._global_best_position.copy(), self._global_best_fitness
