from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class IntrinsicReward:
    type: str
    value: float
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CuriosityDrivenExploration:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self._forward_model_W = np.random.default_rng(42).standard_normal((state_dim + action_dim, hidden_dim))
        self._forward_model_V = np.random.default_rng(43).standard_normal((hidden_dim, state_dim))
        self._inverse_model_W = np.random.default_rng(44).standard_normal((state_dim + state_dim, hidden_dim))
        self._inverse_model_V = np.random.default_rng(45).standard_normal((hidden_dim, action_dim))
        self._error_history: List[float] = []

    def forward_model(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        combined = np.concatenate([state, action])
        h = np.tanh(combined @ self._forward_model_W)
        return h @ self._forward_model_V

    def inverse_model(self, state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        combined = np.concatenate([state, next_state])
        h = np.tanh(combined @ self._inverse_model_W)
        return np.tanh(h @ self._inverse_model_V)

    def intrinsic_reward(self, state: np.ndarray, action: np.ndandom, next_state: np.ndarray) -> float:
        predicted_next = self.forward_model(state, action)
        forward_error = float(np.mean((predicted_next - next_state) ** 2))
        self._error_history.append(forward_error)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]
        reward = min(forward_error * 10.0, 1.0)
        return reward

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        pred_next = self.forward_model(state, action)
        error = pred_next - next_state
        grad_forward_W = np.outer(np.concatenate([state, action]), error @ self._forward_model_V.T)
        grad_forward_V = np.outer(np.tanh(np.concatenate([state, action]) @ self._forward_model_W), error)
        self._forward_model_W -= self.learning_rate * grad_forward_W
        self._forward_model_V -= self.learning_rate * grad_forward_V
        pred_action = self.inverse_model(state, next_state)
        action_error = pred_action - action
        grad_inverse_W = np.outer(np.concatenate([state, next_state]), action_error @ self._inverse_model_V.T)
        grad_inverse_V = np.outer(np.tanh(np.concatenate([state, next_state]) @ self._inverse_model_W), action_error)
        self._inverse_model_W -= self.learning_rate * grad_inverse_W
        self._inverse_model_V -= self.learning_rate * grad_inverse_V

    def get_error_stats(self) -> Dict[str, float]:
        if not self._error_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(self._error_history)),
            "std": float(np.std(self._error_history)),
            "min": float(np.min(self._error_history)),
            "max": float(np.max(self._error_history)),
        }


class RandomNetworkDistillation:
    def __init__(self, state_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001) -> None:
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(42)
        self._target_W = self._rng.standard_normal((state_dim, hidden_dim))
        self._predictor_W = self._rng.standard_normal((state_dim, hidden_dim))
        self._error_history: List[float] = []

    def target_representation(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(state @ self._target_W)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(state @ self._predictor_W)

    def intrinsic_reward(self, state: np.ndarray) -> float:
        target = self.target_representation(state)
        prediction = self.predict(state)
        error = float(np.mean((target - prediction) ** 2))
        self._error_history.append(error)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]
        return min(error * 10.0, 1.0)

    def update(self, state: np.ndarray) -> None:
        target = self.target_representation(state)
        prediction = self.predict(state)
        error = prediction - target
        self._predictor_W -= self.learning_rate * np.outer(state, error * (1 - prediction**2))

    def get_stats(self) -> Dict[str, float]:
        if not self._error_history:
            return {"mean": 0.0, "std": 0.0}
        return {"mean": float(np.mean(self._error_history)), "std": float(np.std(self._error_history))}


class CompetenceBasedMotivation:
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._success_history: List[float] = []
        self._competence_history: List[float] = []
        self._mastery_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        self._current_level = 0

    def record_success(self, success: float) -> None:
        self._success_history.append(success)
        if len(self._success_history) > self.window_size:
            self._success_history = self._success_history[-self.window_size // 2 :]

    def competence(self) -> float:
        if not self._success_history:
            return 0.0
        return float(np.mean(self._success_history[-self.window_size :]))

    def progress(self) -> float:
        if len(self._success_history) < 2:
            return 0.0
        recent = np.mean(self._success_history[-20:])
        older = np.mean(self._success_history[: max(1, len(self._success_history) - 20)])
        return max(0.0, recent - older)

    def mastery_level(self) -> int:
        competence = self.competence()
        for i, threshold in enumerate(self._mastery_thresholds):
            if competence < threshold:
                return i
        return len(self._mastery_thresholds)

    def intrinsic_reward(self) -> float:
        progress = self.progress()
        competence = self.competence()
        challenge = 1.0 - competence
        reward = 0.5 * progress + 0.5 * challenge * progress
        return min(reward, 1.0)

    def get_competence_report(self) -> Dict[str, Any]:
        return {
            "competence": self.competence(),
            "progress": self.progress(),
            "mastery_level": self.mastery_level(),
            "total_episodes": len(self._success_history),
        }


class AutotelicAgent:
    def __init__(self, state_dim: int, action_dim: int, curiosity_weight: float = 0.5, competence_weight: float = 0.5) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.curiosity_weight = curiosity_weight
        self.competence_weight = competence_weight
        self.icm = CuriosityDrivenExploration(state_dim, action_dim)
        self.rnd = RandomNetworkDistillation(state_dim)
        self.competence = CompetenceBasedMotivation()
        self._goal_history: List[Dict[str, Any]] = []

    def intrinsic_motivation(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> IntrinsicReward:
        curiosity_reward = self.icm.intrinsic_reward(state, action, next_state)
        rnd_reward = self.rnd.intrinsic_reward(next_state)
        curiosity = (curiosity_reward + rnd_reward) / 2.0
        success = 1.0 if np.mean(next_state - state) > 0 else 0.0
        self.competence.record_success(success)
        competence_reward = self.competence.intrinsic_reward()
        total = self.curiosity_weight * curiosity + self.competence_weight * competence_reward
        return IntrinsicReward(
            type="autotelic",
            value=total,
            source="curiosity+competence",
            metadata={"curiosity": curiosity, "competence": competence_reward, "success": success},
        )

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        self.icm.update(state, action, next_state)
        self.rnd.update(next_state)

    def set_goal(self, goal: Dict[str, Any]) -> None:
        self._goal_history.append({"goal": goal, "timestamp": time.time()})

    def get_report(self) -> Dict[str, Any]:
        return {
            "curiosity_stats": self.icm.get_error_stats(),
            "rnd_stats": self.rnd.get_stats(),
            "competence_report": self.competence.get_competence_report(),
            "goals_set": len(self._goal_history),
        }


class EmpowermentMaximization:
    def __init__(self, state_dim: int, action_dim: int, horizon: int = 5, num_samples: int = 100) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self._rng = np.random.default_rng(42)
        self._empowerment_history: List[float] = []

    def _random_policy(self, state: np.ndarray) -> np.ndarray:
        return self._rng.standard_normal(self.action_dim)

    def _transition_model(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.tanh(state + 0.5 * action)

    def compute_empowerment(self, state: np.ndarray) -> float:
        futures = []
        for _ in range(self.num_samples):
            s = state.copy()
            for _ in range(self.horizon):
                a = self._random_policy(s)
                s = self._transition_model(s, a)
            futures.append(s.copy())
        entropy = 0.0
        for dim in range(self.state_dim):
            values = [f[dim] for f in futures]
            hist, _ = np.histogram(values, bins=10)
            probs = hist / max(sum(hist), 1e-8)
            probs = probs[probs > 0]
            entropy -= np.sum(probs * np.log(probs + 1e-10))
        empowerment = entropy / self.horizon
        self._empowerment_history.append(empowerment)
        if len(self._empowerment_history) > 500:
            self._empowerment_history = self._empowerment_history[-250:]
        return empowerment

    def empowerment_reward(self, state: np.ndarray) -> float:
        emp = self.compute_empowerment(state)
        return min(emp / 2.0, 1.0)

    def get_history(self) -> List[float]:
        return list(self._empowerment_history)


class PredictionErrorReward:
    def __init__(self, state_dim: int, learning_rate: float = 0.01) -> None:
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self._prediction_model = np.zeros(state_dim)
        self._error_history: List[float] = []

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self._prediction_model

    def prediction_error(self, state: np.ndarray) -> float:
        error = float(np.mean((self._prediction_model - state) ** 2))
        self._error_history.append(error)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]
        return error

    def reward(self, state: np.ndarray) -> float:
        error = self.prediction_error(state)
        return min(error * 10.0, 1.0)

    def update(self, state: np.ndarray) -> None:
        self._prediction_model += self.learning_rate * (state - self._prediction_model)

    def get_stats(self) -> Dict[str, float]:
        if not self._error_history:
            return {"mean": 0.0, "std": 0.0}
        return {"mean": float(np.mean(self._error_history)), "std": float(np.std(self._error_history))}


class IntrinsicMotivation:
    def __init__(self, state_dim: int, action_dim: int, horizon: int = 5) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.icm = CuriosityDrivenExploration(state_dim, action_dim)
        self.rnd = RandomNetworkDistillation(state_dim)
        self.competence = CompetenceBasedMotivation()
        self.autotelic = AutotelicAgent(state_dim, action_dim)
        self.empowerment = EmpowermentMaximization(state_dim, action_dim, horizon=horizon)
        self.prediction_error = PredictionErrorReward(state_dim)
        self._reward_log: List[IntrinsicReward] = []

    def compute_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> IntrinsicReward:
        curiosity = self.icm.intrinsic_reward(state, action, next_state)
        rnd = self.rnd.intrinsic_reward(next_state)
        success = 1.0 if np.mean(next_state - state) > 0 else 0.0
        self.competence.record_success(success)
        competence = self.competence.intrinsic_reward()
        empowerment = self.empowerment.empowerment_reward(state)
        prediction_err = self.prediction_error.reward(state)
        total = (curiosity + rnd + competence + empowerment + prediction_err) / 5.0
        reward = IntrinsicReward(
            type="combined_intrinsic",
            value=total,
            source="icm+rnd+competence+empowerment+prediction_error",
            metadata={
                "curiosity": curiosity,
                "rnd": rnd,
                "competence": competence,
                "empowerment": empowerment,
                "prediction_error": prediction_err,
            },
        )
        self._reward_log.append(reward)
        return reward

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        self.icm.update(state, action, next_state)
        self.rnd.update(next_state)
        self.autotelic.update(state, action, next_state)
        self.prediction_error.update(next_state)

    def get_motivation_report(self) -> Dict[str, Any]:
        return {
            "icm_stats": self.icm.get_error_stats(),
            "rnd_stats": self.rnd.get_stats(),
            "competence": self.competence.get_competence_report(),
            "empowerment_history_size": len(self.empowerment.get_history()),
            "prediction_error_stats": self.prediction_error.get_stats(),
            "total_rewards_computed": len(self._reward_log),
        }
