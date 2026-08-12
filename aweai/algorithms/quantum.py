from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class QubitState:
    amplitudes: np.ndarray
    basis: str = "computational"
    measured: bool = False
    measurement_result: Optional[int] = None


class QuantumSuperpositionSearch:
    def __init__(self, dim: int, num_iterations: int = 10) -> None:
        self.dim = dim
        self.num_iterations = num_iterations
        self._rng = np.random.default_rng(42)

    def initialize_superposition(self) -> np.ndarray:
        psi = np.ones(self.dim, dtype=complex) / math.sqrt(self.dim)
        return psi

    def oracle(self, psi: np.ndarray, marked: Sequence[int]) -> np.ndarray:
        marked_set = set(marked)
        for i in marked_set:
            if 0 <= i < self.dim:
                psi[i] *= -1
        return psi

    def diffusion_operator(self, psi: np.ndarray) -> np.ndarray:
        mean_amp = np.mean(psi)
        return 2 * mean_amp - psi

    def grover_search(self, marked_indices: Sequence[int], max_iterations: Optional[int] = None) -> Tuple[int, float]:
        iterations = max_iterations or self.num_iterations
        psi = self.initialize_superposition()
        for _ in range(iterations):
            psi = self.oracle(psi, marked_indices)
            psi = self.diffusion_operator(psi)
            probs = np.abs(psi) ** 2
        best_idx = int(np.argmax(probs))
        probability = float(probs[best_idx])
        return best_idx, probability

    def amplitude_amplification(self, psi: np.ndarray, grover_iterations: int) -> np.ndarray:
        for _ in range(grover_iterations):
            psi = self.oracle(psi, [])
            psi = self.diffusion_operator(psi)
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        return psi


class QuantumAnnealingSimulator:
    def __init__(self, num_qubits: int, num_steps: int = 100) -> None:
        self.num_qubits = num_qubits
        self.num_steps = num_steps
        self._rng = np.random.default_rng(42)
        self._schedule: List[float] = []

    def transverse_field(self, step: int, total_steps: int) -> float:
        return max(0.0, 1.0 - step / total_steps)

    def problem_hamiltonian(self, state: np.ndarray) -> float:
        return float(np.sum(state**2))

    def anneal(self, initial_state: np.ndarray, cost_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        current = initial_state.copy()
        current_cost = cost_function(current)
        best_state = current.copy()
        best_cost = current_cost
        for step in range(self.num_steps):
            gamma = self.transverse_field(step, self.num_steps)
            proposal = current + self._rng.normal(0, math.sqrt(gamma * 0.1), size=current.shape)
            proposal = np.clip(proposal, -1.0, 1.0)
            proposal_cost = cost_function(proposal)
            delta = proposal_cost - current_cost
            if delta <= 0 or self._rng.random() < math.exp(-delta / max(gamma, 1e-8)):
                current = proposal
                current_cost = proposal_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current.copy()
            self._schedule.append(gamma)
        return best_state, best_cost

    def get_schedule(self) -> List[float]:
        return list(self._schedule)


class QuantumTeleportationChannel:
    def __init__(self, dim: int, fidelity: float = 0.95) -> None:
        self.dim = dim
        self.fidelity = fidelity
        self._rng = np.random.default_rng(42)
        self._noise_std = math.sqrt(1 - fidelity)

    def entangle(self, alice_state: np.ndarray, bob_state: np.ndarray) -> np.ndarray:
        entangled = np.concatenate([alice_state, bob_state]) / math.sqrt(2)
        return entangled

    def measure_bell(self, state: np.ndarray) -> Tuple[int, int]:
        half = len(state) // 2
        alice_part = state[:half]
        bob_part = state[half:]
        alice_measure = self._rng.integers(0, 2)
        bob_measure = self._rng.integers(0, 2)
        return alice_measure, bob_measure

    def teleport(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        entangled_pair = self.entangle(np.ones(self.dim), np.ones(self.dim))
        joint = np.kron(state, entangled_pair)
        alice_meas, bob_meas = self.measure_bell(joint)
        noise = self._rng.normal(0, self._noise_std, size=self.dim)
        received = state + noise
        received = received / (np.linalg.norm(received) + 1e-8)
        fidelity = float(np.abs(np.dot(state, received)) ** 2)
        return received, fidelity


class VariationalQuantumEigensolver:
    def __init__(self, num_qubits: int, ansatz_depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.ansatz_depth = ansatz_depth
        self._rng = np.random.default_rng(42)
        self._params = self._rng.standard_normal(ansatz_depth * num_qubits)
        self._energy_history: List[float] = []

    def ansatz(self, params: np.ndarray, state: np.ndarray) -> np.ndarray:
        for d in range(self.ansatz_depth):
            start = d * self.num_qubits
            rotations = params[start : start + self.num_qubits]
            state = np.cos(rotations) * state + 1j * np.sin(rotations) * np.roll(state, 1)
        return state / (np.linalg.norm(state) + 1e-8)

    def hamiltonian_expectation(self, state: np.ndarray) -> float:
        z_terms = np.sum(np.abs(state) ** 2 * np.arange(len(state)))
        x_terms = float(np.abs(np.sum(state * np.roll(state, 1))) ** 2)
        return float(z_terms - x_terms)

    def energy(self, params: np.ndarray, state: np.ndarray) -> float:
        evolved = self.ansatz(params, state)
        energy = self.hamiltonian_expectation(evolved)
        self._energy_history.append(energy)
        return energy

    def optimize(self, initial_state: np.ndarray, learning_rate: float = 0.01, steps: int = 100) -> Tuple[np.ndarray, float]:
        current_state = initial_state.copy()
        best_params = self._params.copy()
        best_energy = self.energy(best_params, current_state)
        for _ in range(steps):
            grad = self._rng.standard_normal_like(self._params)
            candidate_params = best_params - learning_rate * grad
            energy = self.energy(candidate_params, current_state)
            if energy < best_energy:
                best_energy = energy
                best_params = candidate_params.copy()
        self._params = best_params
        return best_params, best_energy

    def get_energy_history(self) -> List[float]:
        return list(self._energy_history)


class QAOA:
    def __init__(self, num_qubits: int, num_layers: int = 2) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._rng = np.random.default_rng(42)
        self._gammas = self._rng.standard_normal(num_layers)
        self._betas = self._rng.standard_normal(num_layers)

    def cost_hamiltonian(self, bitstring: str) -> float:
        z = np.array([1 if b == "1" else -1 for b in bitstring], dtype=float)
        return float(np.sum(z * np.roll(z, 1)))

    def mixer_hamiltonian(self, state: np.ndarray) -> np.ndarray:
        return np.roll(state, 1) + np.roll(state, -1)

    def apply_layer(self, state: np.ndarray, gamma: float, beta: float) -> np.ndarray:
        cost_phase = np.exp(-1j * gamma * self.cost_hamiltonian("".join("1" if abs(s) > 0.5 else "0" for s in state)))
        mixer_effect = self.mixer_hamiltonian(state)
        state = np.cos(beta) * state + 1j * np.sin(beta) * mixer_effect
        return state / (np.linalg.norm(state) + 1e-8)

    def sample(self, initial_state: np.ndarray) -> str:
        state = initial_state.copy()
        for gamma, beta in zip(self._gammas, self._betas):
            state = self.apply_layer(state, gamma, beta)
        probs = np.abs(state) ** 2
        best_idx = int(np.argmax(probs))
        bitstring = format(best_idx, f"0{self.num_qubits}b")
        return bitstring

    def optimize(self, initial_state: np.ndarray, learning_rate: float = 0.1, steps: int = 50) -> Tuple[np.ndarray, float]:
        best_energy = math.inf
        best_params = np.concatenate([self._gammas, self._betas])
        for _ in range(steps):
            grad = self._rng.standard_normal(len(best_params)) * 0.01
            candidate = best_params - learning_rate * grad
            self._gammas, self._betas = candidate[: self.num_layers], candidate[self.num_layers :]
            bitstring = self.sample(initial_state)
            energy = self.cost_hamiltonian(bitstring)
            if energy < best_energy:
                best_energy = energy
                best_params = candidate.copy()
        self._gammas, self._betas = best_params[: self.num_layers], best_params[self.num_layers :]
        return best_params, best_energy


class QuantumInspiredClassical:
    def __init__(self, dim: int, num_particles: int = 30) -> None:
        self.dim = dim
        self.num_particles = num_particles
        self._rng = np.random.default_rng(42)
        self._particles = self._rng.standard_normal((num_particles, dim))
        self._best_history: List[float] = []

    def quantum_rotation(self, particle: np.ndarray, best: np.ndarray, direction: float = 1.0) -> np.ndarray:
        angle = self._rng.uniform(-math.pi / 4, math.pi / 4) * direction
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        rotated = particle.copy()
        for i in range(0, len(rotated) - 1, 2):
            block = rotated[i : i + 2]
            rotated[i : i + 2] = rotation_matrix @ block
        return rotated

    def quantum_mutation(self, particle: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        mask = self._rng.random(particle.shape) < mutation_rate
        quantum_noise = self._rng.normal(0, 1, size=particle.shape)
        rotated_noise = self.quantum_rotation(quantum_noise, direction=1.0)
        mutated = particle.copy()
        mutated[mask] += rotated_noise[mask] * 0.1
        return mutated

    def evolve(self, fitness_fn: Callable[[np.ndarray], float], generations: int = 100) -> np.ndarray:
        best_global = self._particles[0].copy()
        best_fitness = fitness_fn(best_global)
        for _ in range(generations):
            for i in range(self.num_particles):
                self._particles[i] = self.quantum_mutation(self._particles[i])
                self._particles[i] = self.quantum_rotation(self._particles[i], best_global, direction=1.0)
                fitness = fitness_fn(self._particles[i])
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_global = self._particles[i].copy()
            self._best_history.append(best_fitness)
        return best_global

    def superposition_sampling(self, num_samples: int) -> np.ndarray:
        samples = []
        for _ in range(num_samples):
            idx = self._rng.integers(0, self.num_particles)
            samples.append(self._particles[idx])
        return np.array(samples)
