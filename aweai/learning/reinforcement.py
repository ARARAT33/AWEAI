from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class QLearning:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.01, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(42)
        self.q_table = np.zeros((state_dim, action_dim))
        self._history: List[float] = []

    def act(self, state: int) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.action_dim))
        return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        target = reward + (1 - done) * self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        self._history.append(float(td_error))
        return float(td_error)

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: Any, episodes: int = 100) -> List[float]:
        returns = []
        for _ in range(episodes):
            state = env.reset() if hasattr(env, "reset") else 0
            done = False
            total_reward = 0.0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action) if hasattr(env, "step") else (0, 0, True, {})
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            self.update_epsilon()
            returns.append(total_reward)
        return returns

    def q_values(self) -> np.ndarray:
        return self.q_table.copy()

    def history(self) -> List[float]:
        return list(self._history)


class DQN:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995, batch_size: int = 32, buffer_size: int = 10000) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self._rng = np.random.default_rng(42)
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self._buffer: List[Tuple[Any, ...]] = []
        self._buffer_size = buffer_size
        self._history: List[float] = []

    def _build_model(self) -> Dict[str, np.ndarray]:
        w1 = self._rng.standard_normal((self.state_dim, 64)) * 0.1
        b1 = np.zeros(64)
        w2 = self._rng.standard_normal((64, self.action_dim)) * 0.1
        b2 = np.zeros(self.action_dim)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _forward(self, state: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
        h = np.maximum(state @ model["w1"] + model["b1"], 0.0)
        return h @ model["w2"] + model["b2"]

    def act(self, state: np.ndarray) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.action_dim))
        q_values = self._forward(np.asarray(state, dtype=float).reshape(1, -1), self.q_network)
        return int(np.argmax(q_values))

    def remember(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        if len(self._buffer) >= self._buffer_size:
            self._buffer.pop(0)
        self._buffer.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        if len(self._buffer) < self.batch_size:
            return 0.0
        indices = self._rng.choice(len(self._buffer), size=self.batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            state_arr = np.asarray(state, dtype=float).reshape(1, -1)
            next_state_arr = np.asarray(next_state, dtype=float).reshape(1, -1)
            target = reward + (1 - done) * self.gamma * np.max(self._forward(next_state_arr, self.target_network))
            current_q = self._forward(state_arr, self.q_network)
            current_q[0, action] = target
            h = np.maximum(state_arr @ self.q_network["w1"] + self.q_network["b1"], 0.0)
            error = current_q - (h @ self.q_network["w2"] + self.q_network["b2"])
            self.q_network["w2"] += self.lr * h.T @ error
            self.q_network["b2"] += self.lr * np.sum(error, axis=0)
            total_loss += float(np.mean(error ** 2))
        self._history.append(total_loss / self.batch_size)
        return total_loss / self.batch_size

    def update_target(self) -> None:
        self.target_network = {k: v.copy() for k, v in self.q_network.items()}

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: Any, episodes: int = 100, update_target_every: int = 10) -> List[float]:
        returns = []
        for ep in range(episodes):
            state = env.reset() if hasattr(env, "reset") else np.zeros(self.state_dim)
            done = False
            total_reward = 0.0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action) if hasattr(env, "step") else (np.zeros(self.state_dim), 0.0, True, {})
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                total_reward += reward
            self.update_epsilon()
            if ep % update_target_every == 0:
                self.update_target()
            returns.append(total_reward)
        return returns

    def history(self) -> List[float]:
        return list(self._history)


class PPO:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self._rng = np.random.default_rng(42)
        self.policy = self._build_model()
        self.value = self._build_value_model()
        self._history: List[float] = []

    def _build_model(self) -> Dict[str, np.ndarray]:
        w1 = self._rng.standard_normal((self.state_dim, 64)) * 0.1
        b1 = np.zeros(64)
        w2 = self._rng.standard_normal((64, self.action_dim)) * 0.1
        b2 = np.zeros(self.action_dim)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _build_value_model(self) -> Dict[str, np.ndarray]:
        w1 = self._rng.standard_normal((self.state_dim, 64)) * 0.1
        b1 = np.zeros(64)
        w2 = self._rng.standard_normal((64, 1)) * 0.1
        b2 = np.zeros(1)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _forward_policy(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(state @ self.policy["w1"] + self.policy["b1"], 0.0)
        return np.softmax(h @ self.policy["w2"] + self.policy["b2"], axis=-1)

    def _forward_value(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(state @ self.value["w1"] + self.value["b1"], 0.0)
        return h @ self.value["w2"] + self.value["b2"]

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        probs = self._forward_policy(np.asarray(state, dtype=float).reshape(1, -1))[0]
        action = int(self._rng.choice(self.action_dim, p=probs))
        return action, float(probs[action])

    def compute_gae(self, rewards: Sequence[float], values: Sequence[float], dones: Sequence[bool], lam: float = 0.95) -> np.ndarray:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def update(self, states: np.ndarray, actions: np.ndarray, old_probs: np.ndarray, advantages: np.ndarray, returns: np.ndarray) -> float:
        for _ in range(10):
            new_probs = self._forward_policy(states)
            new_values = self._forward_value(states).flatten()
            action_probs = new_probs[np.arange(len(actions)), actions]
            ratio = action_probs / (old_probs + 1e-8)
            clipped = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -np.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = np.mean((returns - new_values) ** 2)
            loss = policy_loss + 0.5 * value_loss
            self._history.append(float(loss))
        return float(loss)

    def train(self, env: Any, episodes: int = 100, steps_per_update: int = 2048) -> List[float]:
        returns = []
        for _ in range(episodes):
            states, actions, rewards, dones, old_probs = [], [], [], [], []
            state = env.reset() if hasattr(env, "reset") else np.zeros(self.state_dim)
            for _ in range(steps_per_update):
                action, prob = self.act(state)
                next_state, reward, done, _ = env.step(action) if hasattr(env, "step") else (np.zeros(self.state_dim), 0.0, True, {})
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                old_probs.append(prob)
                state = next_state
                if done:
                    break
            values = [self._forward_value(np.asarray(s, dtype=float).reshape(1, -1)).item() for s in states] + [0.0]
            advantages = self.compute_gae(rewards, values, dones)
            returns = advantages + np.array(values[:-1])
            self.update(np.array(states), np.array(actions), np.array(old_probs), advantages, returns)
        return returns.tolist()

    def history(self) -> List[float]:
        return list(self._history)


class SAC:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self._rng = np.random.default_rng(42)
        self.actor = self._build_model((state_dim, action_dim))
        self.critic1 = self._build_model((state_dim + action_dim, 1))
        self.critic2 = self._build_model((state_dim + action_dim, 1))
        self.target_critic1 = self._build_model((state_dim + action_dim, 1))
        self.target_critic2 = self._build_model((state_dim + action_dim, 1))
        self._buffer: List[Tuple[Any, ...]] = []
        self._history: List[float] = []

    def _build_model(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        w1 = self._rng.standard_normal(shape) * 0.1
        b1 = np.zeros(shape[1])
        w2 = self._rng.standard_normal((shape[1], shape[1])) * 0.1
        b2 = np.zeros(shape[1])
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _forward(self, x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
        h = np.maximum(x @ model["w1"] + model["b1"], 0.0)
        return h @ model["w2"] + model["b2"]

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return np.zeros(self.action_dim)
        return self._rng.standard_normal(self.action_dim) * 0.5

    def remember(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 64) -> float:
        if len(self._buffer) < batch_size:
            return 0.0
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        loss = 0.0
        for state, action, reward, next_state, done in batch:
            s = np.asarray(state, dtype=float).reshape(1, -1)
            a = np.asarray(action, dtype=float).reshape(1, -1)
            ns = np.asarray(next_state, dtype=float).reshape(1, -1)
            sa = np.concatenate([s, a], axis=1)
            target_q = reward + (1 - done) * self.gamma * np.min([self._forward(np.concatenate([ns, a], axis=1), self.target_critic1).item(), self._forward(np.concatenate([ns, a], axis=1), self.target_critic2).item()])
            q1 = self._forward(sa, self.critic1).item()
            q2 = self._forward(sa, self.critic2).item()
            loss += ((q1 - target_q) ** 2 + (q2 - target_q) ** 2) / 2.0
        self._history.append(float(loss / batch_size))
        return float(loss / batch_size)

    def update_targets(self) -> None:
        for key in self.target_critic1:
            self.target_critic1[key] = self.tau * self.critic1[key] + (1 - self.tau) * self.target_critic1[key]
            self.target_critic2[key] = self.tau * self.critic2[key] + (1 - self.tau) * self.target_critic2[key]

    def train(self, env: Any, steps: int = 1000) -> List[float]:
        state = env.reset() if hasattr(env, "reset") else np.zeros(self.state_dim)
        for step in range(steps):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action) if hasattr(env, "step") else (np.zeros(self.state_dim), 0.0, True, {})
            self.remember(state, action, reward, next_state, done)
            self.train_step()
            if step % 10 == 0:
                self.update_targets()
            state = next_state
            if done:
                state = env.reset() if hasattr(env, "reset") else np.zeros(self.state_dim)
        return self._history

    def history(self) -> List[float]:
        return list(self._history)
