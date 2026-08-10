"""Auto-scaling with predictive and cost-aware policies."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ScalingPolicy:
    min_replicas: int = 0
    max_replicas: int = 100
    target_gpu_utilization: float = 80.0
    target_cpu_utilization: float = 70.0
    target_queue_depth: int = 5
    scale_up_cooldown: float = 60.0
    scale_down_cooldown: float = 300.0
    scale_up_step: int = 1
    scale_down_step: int = 1
    prediction_window: int = 5
    prediction_threshold: float = 0.7
    scale_to_zero_idle_timeout: float = 600.0
    spot_preemption_risk: float = 0.3
    cold_start_penalty_seconds: float = 120.0
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_gpu_utilization": self.target_gpu_utilization,
            "target_cpu_utilization": self.target_cpu_utilization,
            "target_queue_depth": self.target_queue_depth,
            "scale_up_cooldown": self.scale_up_cooldown,
            "scale_down_cooldown": self.scale_down_cooldown,
            "scale_up_step": self.scale_up_step,
            "scale_down_step": self.scale_down_step,
            "prediction_window": self.prediction_window,
            "prediction_threshold": self.prediction_threshold,
            "scale_to_zero_idle_timeout": self.scale_to_zero_idle_timeout,
            "spot_preemption_risk": self.spot_preemption_risk,
            "cold_start_penalty_seconds": self.cold_start_penalty_seconds,
            "labels": self.labels,
        }


@dataclass
class ScalingEvent:
    timestamp: float
    direction: str
    old_replicas: int
    new_replicas: int
    reason: str
    predicted: bool = False
    cost_delta: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    queue_depth: int = 0


class AutoScaler:
    def __init__(self, policy: Optional[ScalingPolicy] = None) -> None:
        self._policy = policy or ScalingPolicy()
        self._current_replicas: int = 0
        self._history: List[Dict[str, Any]] = []
        self._events: List[ScalingEvent] = []
        self._last_scale_up: float = 0.0
        self._last_scale_down: float = 0.0
        self._idle_since: Optional[float] = None
        self._cost_accumulator: float = 0.0
        self._preempted_count: int = 0
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {
            "scale_up": [],
            "scale_down": [],
            "scale_zero": [],
            "scale_inf": [],
            "preempt": [],
        }
        self._metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._provisioner: Optional[Callable[[int], Dict[str, Any]]] = None

    def set_metrics_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        self._metrics_provider = provider

    def set_provisioner(self, provisioner: Callable[[int], Dict[str, Any]]) -> None:
        self._provisioner = provisioner

    def register_callback(self, event: str, callback: Callable[..., Any]) -> None:
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def evaluate(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not metrics and self._metrics_provider:
            metrics = self._metrics_provider()
        metrics = metrics or {}
        now = time.time()
        gpu_util = metrics.get("gpu_utilization", 0.0)
        cpu_util = metrics.get("cpu_utilization", 0.0)
        queue_depth = metrics.get("queue_depth", 0)
        idle = metrics.get("idle", False)
        predicted_load = self._predict_load(metrics)
        current = self._current_replicas
        result: Dict[str, Any] = {
            "timestamp": now,
            "current_replicas": current,
            "gpu_utilization": gpu_util,
            "cpu_utilization": cpu_util,
            "queue_depth": queue_depth,
            "predicted_load": round(predicted_load, 4),
            "action": "none",
            "target_replicas": current,
        }
        if current == 0 and self._should_scale_up(metrics, predicted_load, now):
            target = self._compute_target_replicas(metrics, predicted_load, from_zero=True)
            target = max(1, min(target, self._policy.max_replicas))
            return self._scale_to(target, "scale_from_zero", now, predicted=True)
        if current > 0 and idle and self._policy.min_replicas == 0:
            idle_duration = now - (self._idle_since or now)
            if idle_duration >= self._policy.scale_to_zero_idle_timeout:
                self._idle_since = None
                return self._scale_to(0, "scale_to_zero", now)
        if idle:
            self._idle_since = self._idle_since or now
        else:
            self._idle_since = None
        if self._should_scale_up(metrics, predicted_load, now):
            target = self._compute_target_replicas(metrics, predicted_load)
            target = min(target, self._policy.max_replicas)
            if target > current:
                return self._scale_to(target, "scale_up", now, predicted=(predicted_load > self._policy.prediction_threshold))
        if self._should_scale_down(metrics, predicted_load, now):
            target = self._compute_target_replicas(metrics, predicted_load)
            target = max(target, self._policy.min_replicas)
            if target < current:
                return self._scale_to(target, "scale_down", now)
        return result

    def _should_scale_up(self, metrics: Dict[str, Any], predicted_load: float, now: float) -> bool:
        if self._current_replicas >= self._policy.max_replicas:
            return False
        if predicted_load > self._policy.prediction_threshold and (now - self._last_scale_up) > self._policy.scale_up_cooldown:
            return True
        gpu_util = metrics.get("gpu_utilization", 0.0)
        cpu_util = metrics.get("cpu_utilization", 0.0)
        queue_depth = metrics.get("queue_depth", 0)
        if gpu_util > self._policy.target_gpu_utilization and (now - self._last_scale_up) > self._policy.scale_up_cooldown:
            return True
        if cpu_util > self._policy.target_cpu_utilization and queue_depth >= self._policy.target_queue_depth:
            return True
        if queue_depth > self._policy.target_queue_depth * 2:
            return True
        return False

    def _should_scale_down(self, metrics: Dict[str, Any], predicted_load: float, now: float) -> bool:
        if self._current_replicas <= self._policy.min_replicas:
            return False
        if predicted_load > self._policy.prediction_threshold:
            return False
        if (now - self._last_scale_down) < self._policy.scale_down_cooldown:
            return False
        gpu_util = metrics.get("gpu_utilization", 0.0)
        cpu_util = metrics.get("cpu_utilization", 0.0)
        queue_depth = metrics.get("queue_depth", 0)
        if gpu_util < self._policy.target_gpu_utilization * 0.4 and queue_depth == 0:
            return True
        if cpu_util < self._policy.target_cpu_utilization * 0.3 and queue_depth == 0:
            return True
        return False

    def _compute_target_replicas(self, metrics: Dict[str, Any], predicted_load: float, from_zero: bool = False) -> int:
        gpu_util = metrics.get("gpu_utilization", 0.0)
        cpu_util = metrics.get("cpu_utilization", 0.0)
        queue_depth = metrics.get("queue_depth", 0)
        current = self._current_replicas
        if gpu_util > 0:
            target_by_gpu = int(math.ceil(gpu_util / self._policy.target_gpu_utilization * current))
        else:
            target_by_gpu = 0
        if cpu_util > 0:
            target_by_cpu = int(math.ceil(cpu_util / self._policy.target_cpu_utilization * current))
        else:
            target_by_cpu = 0
        target_by_queue = max(0, queue_depth)
        if from_zero:
            target = max(target_by_gpu, target_by_cpu, target_by_queue, 1)
        else:
            target = max(target_by_gpu, target_by_cpu, target_by_queue)
        target = max(target, self._policy.min_replicas)
        return min(target, self._policy.max_replicas)

    def _predict_load(self, metrics: Dict[str, Any]) -> float:
        if not self._history:
            return 0.0
        window = self._history[-self._policy.prediction_window:]
        if not window:
            return 0.0
        utilizations = [h.get("gpu_utilization", 0.0) / max(self._policy.target_gpu_utilization, 1.0) for h in window]
        utilizations += [h.get("cpu_utilization", 0.0) / max(self._policy.target_cpu_utilization, 1.0) for h in window]
        queue_depths = [h.get("queue_depth", 0) / max(self._policy.target_queue_depth, 1) for h in window]
        combined = utilizations + queue_depths
        if len(combined) < 3:
            return max(0.0, sum(combined) / len(combined)) if combined else 0.0
        avg = sum(combined) / len(combined)
        recent = combined[-3:]
        trend = sum(recent) / len(recent) - avg
        predicted = max(0.0, avg + trend * 1.5)
        return min(predicted, 2.0)

    def _scale_to(self, target: int, reason: str, now: float, predicted: bool = False) -> Dict[str, Any]:
        old = self._current_replicas
        if reason == "scale_up":
            self._last_scale_up = now
            step = self._policy.scale_up_step
            target = min(old + step, target, self._policy.max_replicas)
        elif reason == "scale_down":
            self._last_scale_down = now
            step = self._policy.scale_down_step
            target = max(old - step, target, self._policy.min_replicas)
        elif reason == "scale_to_zero":
            target = 0
        elif reason == "scale_from_zero":
            target = max(1, target)
        if target == old and reason not in ("scale_to_zero",):
            return {
                "timestamp": now,
                "current_replicas": old,
                "target_replicas": target,
                "action": "none",
                "reason": reason,
            }
        gpu_util = self._history[-1].get("gpu_utilization", 0.0) if self._history else 0.0
        cpu_util = self._history[-1].get("cpu_utilization", 0.0) if self._history else 0.0
        queue_depth = self._history[-1].get("queue_depth", 0) if self._history else 0
        cost_delta = self._estimate_cost_delta(old, target)
        event = ScalingEvent(
            timestamp=now,
            direction=reason,
            old_replicas=old,
            new_replicas=target,
            reason=reason,
            predicted=predicted,
            cost_delta=cost_delta,
            gpu_utilization=gpu_util,
            cpu_utilization=cpu_util,
            queue_depth=queue_depth,
        )
        self._events.append(event)
        self._current_replicas = target
        if reason == "scale_up":
            for cb in self._callbacks["scale_up"]:
                cb(target, old)
        elif reason == "scale_down":
            for cb in self._callbacks["scale_down"]:
                cb(target, old)
        elif reason == "scale_to_zero":
            for cb in self._callbacks["scale_zero"]:
                cb(0, old)
        elif reason == "scale_from_zero":
            for cb in self._callbacks["scale_inf"]:
                cb(target, 0)
        return {
            "timestamp": now,
            "current_replicas": target,
            "old_replicas": old,
            "target_replicas": target,
            "action": reason,
            "reason": reason,
            "predicted": predicted,
            "cost_delta": round(cost_delta, 4),
            "gpu_utilization": gpu_util,
            "cpu_utilization": cpu_util,
            "queue_depth": queue_depth,
        }

    def _estimate_cost_delta(self, old: int, new: int) -> float:
        if self._provisioner:
            try:
                cost = self._provisioner(abs(new - old))
                return cost if new > old else -cost
            except Exception:
                pass
        return float(new - old) * 0.5

    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        self._history.append(dict(metrics, timestamp=time.time()))
        max_history = self._policy.prediction_window * 10
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]

    def handle_preemption(self, preempted_count: int = 1) -> Dict[str, Any]:
        self._preempted_count += preempted_count
        current = self._current_replicas
        target = min(current + preempted_count * 2, self._policy.max_replicas)
        event = ScalingEvent(
            timestamp=time.time(),
            direction="preempt_recover",
            old_replicas=current,
            new_replicas=target,
            reason="spot_preemption",
            cost_delta=0.0,
        )
        self._events.append(event)
        for cb in self._callbacks["preempt"]:
            cb(preempted_count)
        return {"preempted": preempted_count, "target_replicas": target, "action": "preempt_recover"}

    def get_cost_report(self) -> Dict[str, Any]:
        total_cost = sum(abs(e.cost_delta) for e in self._events)
        spot_savings = 0.0
        for e in self._events:
            if e.direction in ("scale_up", "scale_from_zero"):
                spot_savings += e.cost_delta * 0.6
        return {
            "total_cost_delta": round(total_cost, 4),
            "spot_savings_estimate": round(spot_savings, 4),
            "preempted_count": self._preempted_count,
            "scale_events": len(self._events),
            "average_interval_seconds": round(sum(
                (self._events[i].timestamp - self._events[i - 1].timestamp)
                for i in range(1, len(self._events))
            ) / max(len(self._events) - 1, 1), 2) if len(self._events) > 1 else 0.0,
        }

    def get_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        events = self._events[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "direction": e.direction,
                "old_replicas": e.old_replicas,
                "new_replicas": e.new_replicas,
                "reason": e.reason,
                "predicted": e.predicted,
                "cost_delta": e.cost_delta,
                "gpu_utilization": e.gpu_utilization,
                "cpu_utilization": e.cpu_utilization,
                "queue_depth": e.queue_depth,
            }
            for e in events
        ]
