from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Individual:
    genome: np.ndarray
    fitness: float = 0.0
    age: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneticAlgorithm:
    def __init__(self, population_size: int = 100, genome_size: int = 50, mutation_rate: float = 0.01, crossover_rate: float = 0.7) -> None:
        self.population_size = population_size
        self.genome_size = genome_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._rng = np.random.default_rng(42)
        self.population = [Individual(genome=self._rng.standard_normal(genome_size)) for _ in range(population_size)]
        self._generation = 0
        self._best_individual: Optional[Individual] = None
        self._history: List[Dict[str, Any]] = []

    def evaluate(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        for individual in self.population:
            individual.fitness = fitness_fn(individual.genome)
            individual.age += 1
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        if self._best_individual is None or self.population[0].fitness > self._best_individual.fitness:
            self._best_individual = Individual(genome=self.population[0].genome.copy(), fitness=self.population[0].fitness)
        self._history.append({"generation": self._generation, "best_fitness": self.population[0].fitness, "avg_fitness": np.mean([ind.fitness for ind in self.population])})

    def select(self, tournament_size: int = 3) -> Individual:
        tournament = self._rng.choice(self.population, size=min(tournament_size, len(self.population)), replace=False)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if self._rng.random() > self.crossover_rate:
            return Individual(genome=parent1.genome.copy()), Individual(genome=parent2.genome.copy())
        point = self._rng.integers(1, self.genome_size)
        child1_genome = np.concatenate([parent1.genome[:point], parent2.genome[point:]])
        child2_genome = np.concatenate([parent2.genome[:point], parent1.genome[point:]])
        return Individual(genome=child1_genome), Individual(genome=child2_genome)

    def mutate(self, individual: Individual) -> Individual:
        mask = self._rng.random(individual.genome.shape) < self.mutation_rate
        individual.genome[mask] += self._rng.standard_normal(np.sum(mask))
        return individual

    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> Dict[str, Any]:
        self.evaluate(fitness_fn)
        new_population = [self._best_individual] if self._best_individual else []
        while len(new_population) < self.population_size:
            parent1 = self.select()
            parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = new_population[: self.population_size]
        self._generation += 1
        return self._history[-1] if self._history else {}

    def run(self, fitness_fn: Callable[[np.ndarray], float], generations: int = 100) -> List[Dict[str, Any]]:
        for _ in range(generations):
            self.step(fitness_fn)
        return self._history

    def best_genome(self) -> Optional[np.ndarray]:
        if self._best_individual:
            return self._best_individual.genome.copy()
        return None

    def best_fitness(self) -> float:
        return self._best_individual.fitness if self._best_individual else 0.0

    def generation(self) -> int:
        return self._generation

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class Neuroevolution:
    def __init__(self, population_size: int = 50, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 2) -> None:
        self.population_size = population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.genome_size = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
        self.ga = GeneticAlgorithm(population_size=population_size, genome_size=self.genome_size)

    def _genome_to_weights(self, genome: np.ndarray) -> Dict[str, np.ndarray]:
        idx = 0
        w1 = genome[idx : idx + self.input_dim * self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim * self.hidden_dim
        b1 = genome[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        w2 = genome[idx : idx + self.hidden_dim * self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim * self.output_dim
        b2 = genome[idx : idx + self.output_dim]
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def evaluate_genome(self, genome: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        weights = self._genome_to_weights(genome)
        h = np.maximum(X @ weights["w1"] + weights["b1"], 0.0)
        logits = h @ weights["w2"] + weights["b2"]
        preds = np.argmax(logits, axis=1)
        return float(np.mean(preds == y))

    def evolve(self, X: np.ndarray, y: np.ndarray, generations: int = 50) -> Dict[str, Any]:
        def fitness_fn(genome: np.ndarray) -> float:
            return self.evaluate_genome(genome, X, y)
        history = self.ga.run(fitness_fn, generations=generations)
        best_genome = self.ga.best_genome()
        return {"best_fitness": self.ga.best_fitness(), "generation": self.ga.generation(), "history": history, "genome": best_genome}

    def predict(self, genome: np.ndarray, X: np.ndarray) -> np.ndarray:
        weights = self._genome_to_weights(genome)
        h = np.maximum(X @ weights["w1"] + weights["b1"], 0.0)
        logits = h @ weights["w2"] + weights["b2"]
        return np.argmax(logits, axis=1)


class EvolutionStrategy:
    def __init__(self, dim: int = 50, population_size: int = 50, sigma: float = 0.1, learning_rate: float = 0.01) -> None:
        self.dim = dim
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(42)
        self.theta = self._rng.standard_normal(dim)

    def ask(self) -> np.ndarray:
        perturbations = self._rng.standard_normal((self.population_size, self.dim))
        return self.theta + self.sigma * perturbations

    def tell(self, perturbations: np.ndarray, rewards: np.ndarray) -> None:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        self.theta += self.learning_rate / (self.population_size * self.sigma) * (perturbations.T @ rewards)

    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> float:
        perturbations = self.ask()
        rewards = np.array([fitness_fn(theta) for theta in perturbations])
        self.tell(perturbations, rewards)
        return float(rewards.mean())

    def run(self, fitness_fn: Callable[[np.ndarray], float], iterations: int = 100) -> List[float]:
        history = []
        for _ in range(iterations):
            mean_reward = self.step(fitness_fn)
            history.append(mean_reward)
        return history

    def parameters(self) -> np.ndarray:
        return self.theta.copy()
