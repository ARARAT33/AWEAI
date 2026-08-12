from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class QuantumReasoner:
    def __init__(self, num_qubits: int = 4) -> None:
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self._rng = np.random.default_rng(42)
        self.state = self._rng.standard_normal(self.dim) + 1j * self._rng.standard_normal(self.dim)
        self.state = self.state / np.linalg.norm(self.state)
        self._history: List[np.ndarray] = [self.state.copy()]
        self._measurement_history: List[int] = []

    def apply_hadamard(self, target: int) -> None:
        h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_single_qubit_gate(h, target)

    def apply_pauli_x(self, target: int) -> None:
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_single_qubit_gate(x, target)

    def apply_pauli_y(self, target: int) -> None:
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_single_qubit_gate(y, target)

    def apply_pauli_z(self, target: int) -> None:
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_single_qubit_gate(z, target)

    def apply_cnot(self, control: int, target: int) -> None:
        cnot = np.eye(self.dim, dtype=complex)
        for i in range(self.dim):
            bits = [(i >> (self.num_qubits - 1 - j)) & 1 for j in range(self.num_qubits)]
            if bits[control] == 1:
                bits[target] = 1 - bits[target]
                j_prime = sum(bit << (self.num_qubits - 1 - j) for j, bit in enumerate(bits))
                cnot[i, i] = 0
                cnot[i, j_prime] = 1
        self.state = cnot @ self.state
        self._history.append(self.state.copy())

    def apply_phase(self, target: int, theta: float) -> None:
        p = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
        self._apply_single_qubit_gate(p, target)

    def apply_rotation(self, target: int, axis: str, angle: float) -> None:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        if axis == "x":
            r = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        elif axis == "y":
            r = np.array([[c, -s], [s, c]], dtype=complex)
        else:
            r = np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=complex)
        self._apply_single_qubit_gate(r, target)

    def _apply_single_qubit_gate(self, gate: np.ndarray, target: int) -> None:
        full_gate = np.eye(self.dim, dtype=complex)
        for i in range(0, self.dim, 2):
            bit = (i >> (self.num_qubits - 1 - target)) & 1
            if bit == 0:
                j = i + (1 << (self.num_qubits - 1 - target))
                full_gate[i : i + 2, i : i + 2] = gate
        self.state = full_gate @ self.state
        self._history.append(self.state.copy())

    def measure(self, target: Optional[int] = None) -> int:
        if target is None:
            probs = np.abs(self.state) ** 2
            probs = probs / np.sum(probs)
            outcome = int(self._rng.choice(self.dim, p=probs))
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[outcome] = 1.0
            self._measurement_history.append(outcome)
            return outcome
        reduced = np.zeros(2, dtype=complex)
        for i in range(self.dim):
            bit = (i >> (self.num_qubits - 1 - target)) & 1
            reduced[bit] += self.state[i]
        probs = np.abs(reduced) ** 2
        probs = probs / np.sum(probs)
        outcome = int(self._rng.choice(2, p=probs))
        for i in range(self.dim):
            bit = (i >> (self.num_qubits - 1 - target)) & 1
            if bit != outcome:
                self.state[i] = 0.0
        self.state = self.state / (np.linalg.norm(self.state) + 1e-10)
        self._measurement_history.append(outcome)
        return outcome

    def superposition(self, amplitudes: Sequence[complex]) -> None:
        self.state = np.array(amplitudes, dtype=complex)
        self.state = self.state / np.linalg.norm(self.state)
        self._history.append(self.state.copy())

    def interference(self, other_state: np.ndarray) -> np.ndarray:
        other = np.asarray(other_state, dtype=complex)
        if other.shape != self.state.shape:
            other = np.resize(other, self.state.shape)
        return self.state + other

    def entanglement_measure(self) -> float:
        if self.num_qubits < 2:
            return 0.0
        reshaped = self.state.reshape([2] * self.num_qubits)
        entropy = 0.0
        for i in range(2 ** (self.num_qubits - 1)):
            reduced = reshaped[i]
            probs = np.abs(reduced) ** 2
            probs = probs[probs > 1e-10]
            entropy -= np.sum(probs * np.log2(probs))
        return float(entropy)

    def quantum_probability(self, observable: np.ndarray) -> float:
        obs = np.asarray(observable, dtype=complex)
        if obs.shape != (self.dim, self.dim):
            obs = np.eye(self.dim, dtype=complex)
        expectation = float(np.real(np.conj(self.state) @ obs @ self.state))
        return expectation

    def grover_search(self, oracle: Callable[[int], bool], iterations: Optional[int] = None) -> int:
        if iterations is None:
            iterations = int(math.pi / 4 * math.sqrt(self.dim))
        for _ in range(iterations):
            for i in range(self.dim):
                if oracle(i):
                    self.state[i] = -self.state[i]
            reflected = 2 * np.mean(np.abs(self.state) ** 2) * np.ones(self.dim, dtype=complex) - self.state
            self.state = reflected / np.linalg.norm(reflected)
        probs = np.abs(self.state) ** 2
        return int(np.argmax(probs))

    def state_history(self) -> List[np.ndarray]:
        return list(self._history)

    def measurement_history(self) -> List[int]:
        return list(self._measurement_history)

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2
