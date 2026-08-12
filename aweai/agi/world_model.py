from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class WorldModel:
    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 32) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self._rng = np.random.default_rng(42)
        self.transition_model = self._rng.standard_normal((state_dim + action_dim, state_dim)) * 0.01
        self.reward_model = self._rng.standard_normal((state_dim + action_dim, 1)) * 0.01
        self._history: List[Dict[str, Any]] = []

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        sa = np.concatenate([np.asarray(state, dtype=float).flatten(), np.asarray(action, dtype=float).flatten()])
        return sa @ self.transition_model

    def predict_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        sa = np.concatenate([np.asarray(state, dtype=float).flatten(), np.asarray(action, dtype=float).flatten()])
        return float((sa @ self.reward_model).item())

    def imagine(self, initial_state: np.ndarray, action_sequence: Sequence[np.ndarray]) -> List[np.ndarray]:
        states = [np.asarray(initial_state, dtype=float).flatten()]
        for action in action_sequence:
            next_state = self.predict_next_state(states[-1], np.asarray(action, dtype=float).flatten())
            states.append(next_state)
        return states

    def train(self, transitions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]], lr: float = 0.01) -> Dict[str, float]:
        total_transition_loss = 0.0
        total_reward_loss = 0.0
        for state, action, next_state, reward, done in transitions:
            sa = np.concatenate([np.asarray(state, dtype=float).flatten(), np.asarray(action, dtype=float).flatten()])
            predicted_next = sa @ self.transition_model
            predicted_reward = float((sa @ self.reward_model).item())
            transition_error = np.asarray(next_state, dtype=float).flatten() - predicted_next
            reward_error = reward - predicted_reward
            self.transition_model += lr * np.outer(sa, transition_error)
            self.reward_model += lr * np.outer(sa, np.array([reward_error]))
            total_transition_loss += float(np.mean(transition_error ** 2))
            total_reward_loss += float(reward_error ** 2)
        n = max(len(transitions), 1)
        return {"transition_loss": total_transition_loss / n, "reward_loss": total_reward_loss / n}

    def counterfactual(self, state: np.ndarray, action: np.ndarray, alternative_action: np.ndarray) -> Dict[str, Any]:
        actual_next = self.predict_next_state(state, action)
        actual_reward = self.predict_reward(state, action)
        alt_next = self.predict_next_state(state, alternative_action)
        alt_reward = self.predict_reward(state, alternative_action)
        return {
            "actual_next_state": actual_next.tolist(),
            "actual_reward": actual_reward,
            "alternative_next_state": alt_next.tolist(),
            "alternative_reward": alt_reward,
            "difference_state": (actual_next - alt_next).tolist(),
            "difference_reward": actual_reward - alt_reward,
        }

    def plan(self, initial_state: np.ndarray, horizon: int = 10, num_candidates: int = 10) -> Tuple[np.ndarray, float]:
        best_sequence = None
        best_return = -1e10
        for _ in range(num_candidates):
            actions = [self._rng.standard_normal(self.action_dim) for _ in range(horizon)]
            states = self.imagine(initial_state, actions)
            rewards = [self.predict_reward(states[i], actions[i]) for i in range(horizon)]
            total_return = sum(rewards)
            if total_return > best_return:
                best_return = total_return
                best_sequence = actions[0]
        return best_sequence if best_sequence is not None else np.zeros(self.action_dim), best_return

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class IntrinsicMotivation:
    def __init__(self, state_dim: int = 8, action_dim: int = 4) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(42)
        self.forward_model = self._rng.standard_normal((state_dim + action_dim, state_dim)) * 0.01
        self.inverse_model = self._rng.standard_normal((state_dim + state_dim, action_dim)) * 0.01
        self._visited_states: List[np.ndarray] = []
        self._intrinsic_rewards: List[float] = []

    def curiosity_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        sa = np.concatenate([np.asarray(state, dtype=float).flatten(), np.asarray(action, dtype=float).flatten()])
        predicted_next = sa @ self.forward_model
        error = np.linalg.norm(np.asarray(next_state, dtype=float).flatten() - predicted_next)
        reward = float(error)
        self._intrinsic_rewards.append(reward)
        return reward

    def train_models(self, transitions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], lr: float = 0.01) -> Dict[str, float]:
        total_forward_loss = 0.0
        total_inverse_loss = 0.0
        for state, action, next_state in transitions:
            sa = np.concatenate([np.asarray(state, dtype=float).flatten(), np.asarray(action, dtype=float).flatten()])
            predicted_next = sa @ self.forward_model
            forward_error = np.asarray(next_state, dtype=float).flatten() - predicted_next
            self.forward_model += lr * np.outer(sa, forward_error)
            total_forward_loss += float(np.mean(forward_error ** 2))
            ss = np.concatenate([np.asarray(state, dtype=float).flatten(), np.asarray(next_state, dtype=float).flatten()])
            predicted_action = ss @ self.inverse_model
            inverse_error = np.asarray(action, dtype=float).flatten() - predicted_action
            self.inverse_model += lr * np.outer(ss, inverse_error)
            total_inverse_loss += float(np.mean(inverse_error ** 2))
        n = max(len(transitions), 1)
        return {"forward_loss": total_forward_loss / n, "inverse_loss": total_inverse_loss / n}

    def empowerment(self, state: np.ndarray, num_actions: int = 10) -> float:
        actions = [self._rng.standard_normal(self.action_dim) for _ in range(num_actions)]
        next_states = []
        for action in actions:
            sa = np.concatenate([np.asarray(state, dtype=float).flatten(), action])
            next_state = sa @ self.forward_model
            next_states.append(next_state)
        if len(next_states) < 2:
            return 0.0
        diffs = []
        for i in range(len(next_states)):
            for j in range(i + 1, len(next_states)):
                diffs.append(np.linalg.norm(next_states[i] - next_states[j]))
        return float(np.mean(diffs)) if diffs else 0.0

    def novelty(self, state: np.ndarray) -> float:
        if not self._visited_states:
            return 1.0
        distances = [np.linalg.norm(state - s) for s in self._visited_states]
        min_dist = min(distances)
        return 1.0 / (1.0 + min_dist)

    def update_visited(self, state: np.ndarray) -> None:
        self._visited_states.append(np.asarray(state, dtype=float).flatten())

    def intrinsic_rewards_history(self) -> List[float]:
        return list(self._intrinsic_rewards)
