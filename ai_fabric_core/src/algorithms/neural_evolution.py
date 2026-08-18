"""
Neural Architecture Evolution (NAE) Algorithm
Ավտոմատ նեյրոնային ճարտարապետության էվոլյուցիա

Automated neural architecture search using evolutionary algorithms.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Architecture:
    """Represents a neural network architecture"""
    id: str
    layers: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[tuple] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0


class NeuralArchitectureEvolution:
    """
    Neural Architecture Evolution for automated model design
    
    Features:
    - Genetic algorithm-based search
    - Multi-objective optimization
    - Progressive evolution strategy
    - Fitness-based selection
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        num_generations: int = 100,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        
        self.logger = logging.getLogger(__name__)
        self.population: List[Architecture] = []
        self.best_architecture: Optional[Architecture] = None
        self.history: List[Dict[str, Any]] = []
    
    def initialize_population(self) -> None:
        """Initialize random population"""
        self.population = []
        for i in range(self.population_size):
            arch = self._generate_random_architecture(f"arch_{i}")
            self.population.append(arch)
    
    def _generate_random_architecture(self, arch_id: str) -> Architecture:
        """Generate random architecture"""
        num_layers = np.random.randint(3, 20)
        layers = []
        
        for i in range(num_layers):
            layer_type = np.random.choice(['conv', 'dense', 'attention', 'pool'])
            layer = {
                'type': layer_type,
                'units': np.random.randint(32, 1024),
                'activation': np.random.choice(['relu', 'gelu', 'swish']),
            }
            layers.append(layer)
        
        return Architecture(id=arch_id, layers=layers)
    
    def evolve(self, fitness_fn) -> Architecture:
        """Run evolution process"""
        self.initialize_population()
        
        for gen in range(self.num_generations):
            # Evaluate fitness
            for arch in self.population:
                arch.fitness = fitness_fn(arch)
            
            # Select parents
            parents = self._tournament_selection()
            
            # Create next generation
            next_gen = []
            
            # Elitism - keep best
            best = max(self.population, key=lambda a: a.fitness)
            next_gen.append(best)
            
            while len(next_gen) < self.population_size:
                # Crossover
                if np.random.random() < self.crossover_rate:
                    p1, p2 = np.random.choice(parents, 2, replace=False)
                    child = self._crossover(p1, p2)
                else:
                    child = self._copy_architecture(np.random.choice(parents))
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                child.generation = gen + 1
                next_gen.append(child)
            
            self.population = next_gen
            
            # Track best
            current_best = max(self.population, key=lambda a: a.fitness)
            if self.best_architecture is None or current_best.fitness > self.best_architecture.fitness:
                self.best_architecture = current_best
            
            self.history.append({
                'generation': gen,
                'best_fitness': current_best.fitness,
                'avg_fitness': np.mean([a.fitness for a in self.population]),
            })
            
            self.logger.info(f"Generation {gen}: Best fitness = {current_best.fitness:.4f}")
        
        return self.best_architecture
    
    def _tournament_selection(self, k: int = 5) -> List[Architecture]:
        """Tournament selection"""
        selected = []
        for _ in range(len(self.population)):
            candidates = np.random.choice(self.population, k, replace=False)
            winner = max(candidates, key=lambda a: a.fitness)
            selected.append(winner)
        return selected
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Single-point crossover"""
        point = min(len(parent1.layers), len(parent2.layers)) // 2
        
        child_layers = parent1.layers[:point] + parent2.layers[point:]
        
        return Architecture(
            id=f"child_{parent1.id}_{parent2.id}",
            layers=child_layers,
            generation=max(parent1.generation, parent2.generation) + 1,
        )
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """Mutate architecture"""
        mutated = self._copy_architecture(architecture)
        
        mutation_type = np.random.choice(['add_layer', 'remove_layer', 'modify_layer'])
        
        if mutation_type == 'add_layer' and len(mutated.layers) < 50:
            new_layer = {
                'type': np.random.choice(['conv', 'dense', 'attention']),
                'units': np.random.randint(32, 1024),
            }
            pos = np.random.randint(0, len(mutated.layers) + 1)
            mutated.layers.insert(pos, new_layer)
        
        elif mutation_type == 'remove_layer' and len(mutated.layers) > 2:
            pos = np.random.randint(0, len(mutated.layers))
            mutated.layers.pop(pos)
        
        elif mutation_type == 'modify_layer' and len(mutated.layers) > 0:
            pos = np.random.randint(0, len(mutated.layers))
            mutated.layers[pos]['units'] = np.random.randint(32, 1024)
        
        return mutated
    
    def _copy_architecture(self, arch: Architecture) -> Architecture:
        """Deep copy architecture"""
        return Architecture(
            id=arch.id,
            layers=[l.copy() for l in arch.layers],
            connections=arch.connections.copy(),
            fitness=arch.fitness,
            generation=arch.generation,
        )
    
    def get_best_architecture(self) -> Optional[Architecture]:
        """Get best architecture found"""
        return self.best_architecture
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get evolution history"""
        return self.history
