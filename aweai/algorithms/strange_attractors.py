from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AttractorParameters:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    name: str = "lorenz"


class StrangeAttractor:
    def __init__(self, params: Optional[AttractorParameters] = None) -> None:
        self.params = params or AttractorParameters()
        self._rng = np.random.default_rng(42)
        self._trajectory: List[np.ndarray] = []
        self._lyapunov: Optional[float] = None

    def lorenz(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def rossler(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        a, b, c = 0.2, 0.2, 5.7
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def aizawa(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
        return np.array([(z - b) * x - d * y, d * x + (z - b) * y, c + a * z - z ** 3 / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * x ** 3])

    def thomas(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        b = 0.208186
        return np.array([math.sin(y) - b * x, math.sin(z) - b * y, math.sin(x) - b * z])

    def halvorsen(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        a = 1.89
        return np.array([-a * x - 4 * y - 4 * z - y ** 2, -a * y - 4 * z - 4 * x - z ** 2, -a * z - 4 * x - 4 * y - x ** 2])

    def dadras(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        p, q, r, s, e = 3, 2.7, 1.7, 2, 2
        return np.array([y - p * x + q * y * z, r * z - y * z, s * x * y - e * z])

    def chen(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        a, b, c = 5.0, -10.0, -3.8
        return np.array([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z])

    def generate(self, initial: Optional[np.ndarray] = None, steps: int = 10000, transient: int = 1000) -> np.ndarray:
        state = initial if initial is not None else np.array([1.0, 1.0, 1.0]) + self._rng.standard_normal(3) * 0.1
        trajectory = []
        for i in range(steps + transient):
            state = state + self.params.dt * self.lorenz(state)
            if i >= transient:
                trajectory.append(state.copy())
        self._trajectory = trajectory
        return np.array(trajectory)

    def generate_rossler(self, initial: Optional[np.ndarray] = None, steps: int = 10000, transient: int = 1000) -> np.ndarray:
        state = initial if initial is not None else np.array([1.0, 1.0, 1.0]) + self._rng.standard_normal(3) * 0.1
        trajectory = []
        for i in range(steps + transient):
            state = state + self.params.dt * self.rossler(state)
            if i >= transient:
                trajectory.append(state.copy())
        self._trajectory = trajectory
        return np.array(trajectory)

    def lyapunov_exponent(self, trajectory: Optional[np.ndarray] = None, steps: int = 5000) -> float:
        if trajectory is None:
            trajectory = np.array(self._trajectory)
        if len(trajectory) < 2:
            return 0.0
        distances = [np.linalg.norm(trajectory[i + 1] - trajectory[i]) for i in range(min(steps, len(trajectory) - 1))]
        distances = [d for d in distances if d > 1e-10]
        if not distances:
            return 0.0
        log_distances = np.log(np.array(distances))
        self._lyapunov = float(np.mean(log_distances) / self.params.dt)
        return self._lyapunov

    def fractal_dimension(self, trajectory: Optional[np.ndarray] = None) -> float:
        if trajectory is None:
            trajectory = np.array(self._trajectory)
        if len(trajectory) < 10:
            return 0.0
        scales = np.logspace(0, 3, 20, base=2).astype(int)
        counts = []
        for scale in scales:
            box_counts = 0
            for i in range(0, len(trajectory), scale):
                box_counts += 1
            counts.append(box_counts)
        counts = np.array(counts)
        scales = np.array(scales, dtype=float)
        if np.any(counts == 0) or np.any(scales == 0):
            return 0.0
        log_counts = np.log(counts)
        log_scales = np.log(scales)
        coeffs = np.polyfit(log_scales, log_counts, 1)
        return float(-coeffs[0])

    def trajectory(self) -> List[np.ndarray]:
        return list(self._trajectory)

    def lyapunov(self) -> Optional[float]:
        return self._lyapunov
