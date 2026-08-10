"""Unified API router with fallback chains, A/B testing, cost tracking, and warm pools."""

from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from aweai.compat.providers import BaseProvider, get_provider
from aweai.compat.types import ChatMessage, ChatRequest, ChatResponse, APIError, ModelNotFoundError


@dataclass
class RouteConfig:
    provider: str
    model: Optional[str] = None
    weight: float = 1.0
    fallback: Optional[List[str]] = None
    max_retries: int = 2
    timeout: float = 120.0
    priority: int = 0


@dataclass
class RouteStats:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    last_used: float = 0.0


@dataclass
class ABTestConfig:
    name: str
    variants: Dict[str, RouteConfig]
    traffic_split: Dict[str, float] = field(default_factory=dict)
    metric: str = "latency_ms"
    active: bool = True


class Router:
    def __init__(self) -> None:
        self.routes: Dict[str, RouteConfig] = {}
        self.providers: Dict[str, BaseProvider] = {}
        self.stats: Dict[str, RouteStats] = {}
        self._lock = threading.RLock()
        self.default_model = "local-model"
        self.cost_per_1k_tokens: Dict[str, float] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.warm_pools: Dict[str, Any] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}

    def register(self, name: str, config: RouteConfig) -> None:
        self.routes[name] = config
        self.stats[name] = RouteStats()

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        self.providers[name] = provider

    def set_cost(self, provider: str, cost_per_1k: float) -> None:
        self.cost_per_1k_tokens[provider] = cost_per_1k

    def route(self, request: ChatRequest) -> Tuple[str, RouteConfig]:
        model = request.model
        for name, config in self.routes.items():
            if config.model == model or (model.startswith(config.provider) and config.model is None):
                return name, config
        return "default", RouteConfig(provider="local", model=self.default_model)

    def send(self, request: ChatRequest) -> ChatResponse:
        route_name, config = self.route(request)
        fallbacks = config.fallback or []
        candidates = [config.provider] + [f for f in fallbacks if f != config.provider]
        last_error: Optional[Exception] = None
        for provider_name in candidates:
            try:
                provider = self.providers.get(provider_name)
                if provider is None:
                    provider = get_provider(provider_name, model=config.model or self.default_model)
                    self.providers[provider_name] = provider
                start = time.time()
                response = provider.chat(
                    messages=[m.to_dict() for m in request.messages],
                    max_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.7,
                    **self._build_kwargs(request),
                )
                latency = (time.time() - start) * 1000
                self._record_success(route_name, provider_name, response, latency)
                return response
            except (APIError, RuntimeError) as e:
                last_error = e
                self._record_failure(route_name, provider_name)
                continue
        if last_error:
            raise last_error
        raise ModelNotFoundError(f"No provider found for model {request.model}")

    def send_with_ab(self, request: ChatRequest, ab_test_name: str) -> ChatResponse:
        ab = self.ab_tests.get(ab_test_name)
        if not ab or not ab.active:
            return self.send(request)
        variant = self._pick_variant(ab)
        if variant:
            request.metadata["ab_test"] = ab_test_name
            request.metadata["variant"] = variant
        return self.send(request)

    def _pick_variant(self, ab: ABTestConfig) -> str:
        r = random.random()
        cumulative = 0.0
        for variant, weight in ab.traffic_split.items():
            cumulative += weight
            if r <= cumulative:
                return variant
        return list(ab.traffic_split.keys())[0]

    def get_stats(self, route_name: str) -> Optional[RouteStats]:
        return self.stats.get(route_name)

    def total_cost(self) -> float:
        total = 0.0
        for route_name, stat in self.stats.items():
            cost = self.cost_per_1k_tokens.get(route_name, 0.0)
            total += (stat.total_tokens / 1000.0) * cost
        return total

    def _record_success(self, route_name: str, provider_name: str, response: ChatResponse, latency_ms: float) -> None:
        with self._lock:
            if route_name not in self.stats:
                self.stats[route_name] = RouteStats()
            self.stats[route_name].requests += 1
            self.stats[route_name].successes += 1
            self.stats[route_name].total_latency_ms += latency_ms
            self.stats[route_name].last_used = time.time()
            if response.usage:
                self.stats[route_name].total_tokens += response.usage.total_tokens
            cost_rate = self.cost_per_1k_tokens.get(provider_name, 0.0)
            if response.usage:
                self.stats[route_name].total_cost += (response.usage.total_tokens / 1000.0) * cost_rate

    def _record_failure(self, route_name: str, provider_name: str) -> None:
        with self._lock:
            if route_name not in self.stats:
                self.stats[route_name] = RouteStats()
            self.stats[route_name].requests += 1
            self.stats[route_name].failures += 1

    def _build_kwargs(self, request: ChatRequest) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if request.stop:
            kwargs["stop"] = request.stop
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.functions:
            kwargs["functions"] = [f.to_dict() for f in request.functions]
        if request.function_call is not None:
            kwargs["function_call"] = request.function_call
        if request.tools:
            kwargs["tools"] = [t.to_dict() for t in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format:
            kwargs["response_format"] = request.response_format
        return kwargs

    def warm_pool(self, model: str, count: int = 1) -> None:
        self.warm_pools[model] = count

    def set_rate_limit(self, provider: str, max_rps: int) -> None:
        self.rate_limits[provider] = []

    def check_rate_limit(self, provider: str) -> bool:
        now = time.time()
        with self._lock:
            timestamps = self.rate_limits.get(provider, [])
            timestamps = [t for t in timestamps if now - t < 1.0]
            self.rate_limits[provider] = timestamps
            return len(timestamps) < 100


router = Router()
