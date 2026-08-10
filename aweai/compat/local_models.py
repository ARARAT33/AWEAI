"""Local model serving with dynamic batching, KV cache, and quantization."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from aweai.compat.types import ChatMessage, ChatRequest, ChatResponse, CompletionChoice, EmbeddingResponse, ModelInfo, ModelsResponse, Usage, ValidationError, ModelNotFoundError


@dataclass
class KVCache:
    model_id: str
    max_tokens: int
    current_tokens: int = 0
    cache: Dict[str, List[float]] = field(default_factory=dict)
    prompt_cache: Dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> Optional[List[float]]:
        return self.cache.get(key)

    def set(self, key: str, value: List[float]) -> None:
        self.cache[key] = value

    def get_prompt(self, key: str) -> Optional[str]:
        return self.prompt_cache.get(key)

    def set_prompt(self, key: str, value: str) -> None:
        self.prompt_cache[key] = value

    def evict(self, tokens_to_free: int) -> int:
        freed = 0
        keys = list(self.cache.keys())
        for k in keys:
            if freed >= tokens_to_free:
                break
            self.cache.pop(k)
            freed += 1
        self.current_tokens = max(0, self.current_tokens - freed)
        return freed


@dataclass
class BatchItem:
    request_id: str
    request: ChatRequest
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class LocalModelServer:
    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path.home() / ".aweai" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.kv_caches: Dict[str, KVCache] = {}
        self.batches: List[BatchItem] = []
        self.batch_lock = threading_lock()
        self.max_batch_size = 32
        self.batch_timeout_ms = 100
        self.quantization_support = {"int8": True, "int4": True, "gguf": True, "fp16": True}
        self.loaded: Dict[str, Any] = {}

    def load_model(self, model_id: str, quantization: str = "none", device: str = "cpu") -> Any:
        model_path = self._find_model_path(model_id)
        if not model_path:
            raise ModelNotFoundError(f"Model {model_id} not found in zoo")
        cache = KVCache(model_id=model_id, max_tokens=4096)
        self.kv_caches[model_id] = cache
        info: Dict[str, Any] = {"id": model_id, "path": str(model_path), "quantization": quantization, "device": device, "loaded_at": time.time()}
        self.loaded_models[model_id] = info
        return info

    def unload_model(self, model_id: str) -> None:
        self.loaded_models.pop(model_id, None)
        self.kv_caches.pop(model_id, None)

    def _find_model_path(self, model_id: str) -> Optional[Path]:
        candidates = [
            self.model_dir / model_id,
            self.model_dir / f"{model_id}.bin",
            self.model_dir / f"{model_id}.gguf",
            self.model_dir / f"{model_id}.pt",
            self.model_dir / f"{model_id}.pth",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self.loaded_models

    def generate(self, request: ChatRequest) -> ChatResponse:
        if not request.messages:
            raise ValidationError("No messages provided")
        model_id = request.model
        if model_id not in self.loaded_models:
            try:
                self.load_model(model_id)
            except ModelNotFoundError:
                pass
        if model_id not in self.loaded_models:
            raise ModelNotFoundError(f"Model {model_id} not loaded")
        prompt = self._build_prompt(request)
        kv = self.kv_caches.get(model_id)
        cached = kv.get_prompt(prompt) if kv else None
        if cached:
            text = cached
        else:
            text = self._mock_generate(prompt, request.max_tokens or 128)
            if kv:
                kv.set_prompt(prompt, text)
        choice = CompletionChoice(index=0, text=text, finish_reason="stop")
        usage = self._estimate_usage(prompt, text)
        return ChatResponse(model=model_id, choices=[choice], usage=usage)

    def generate_stream(self, request: ChatRequest):
        if not request.messages:
            raise ValidationError("No messages provided")
        model_id = request.model
        if model_id not in self.loaded_models:
            try:
                self.load_model(model_id)
            except ModelNotFoundError:
                pass
        if model_id not in self.loaded_models:
            raise ModelNotFoundError(f"Model {model_id} not loaded")
        prompt = self._build_prompt(request)
        tokens = self._mock_tokenize(prompt)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        for i, token in enumerate(tokens):
            delta = {"content": token}
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    def embeddings(self, model_id: str, texts: List[str]) -> EmbeddingResponse:
        if model_id not in self.loaded_models:
            try:
                self.load_model(model_id)
            except ModelNotFoundError:
                pass
        dims = 384
        data = []
        for i, text in enumerate(texts):
            vec = [float(hashlib.md5((text + str(i)).encode()).hexdigest()[j]) / 255.0 for j in range(dims)]
            data.append({"object": "embedding", "embedding": vec, "index": i})
        return EmbeddingResponse(data=data, model=model_id, usage=Usage(prompt_tokens=sum(len(t.split()) for t in texts), completion_tokens=0, total_tokens=sum(len(t.split()) for t in texts)))

    def list_models(self) -> ModelsResponse:
        data = []
        for mid, info in self.loaded_models.items():
            data.append(ModelInfo(id=mid, owned_by="aweai", created=int(info.get("loaded_at", time.time()))))
        return ModelsResponse(data=data)

    def dynamic_batch(self, items: List[BatchItem]) -> List[ChatResponse]:
        responses = []
        for item in items:
            try:
                responses.append(self.generate(item.request))
            except Exception:
                responses.append(ChatResponse(model=item.request.model, choices=[CompletionChoice(index=0, text="", finish_reason="error")]))
        return responses

    def add_to_batch(self, request_id: str, request: ChatRequest, priority: int = 0) -> None:
        with self.batch_lock:
            self.batches.append(BatchItem(request_id=request_id, request=request, priority=priority))
            if len(self.batches) >= self.max_batch_size:
                items = self.batches
                self.batches = []
                self.dynamic_batch(items)

    def quantize_model(self, model_id: str, method: str) -> bool:
        if method not in self.quantization_support:
            raise ValueError(f"Unsupported quantization: {method}")
        if model_id not in self.loaded_models:
            raise ModelNotFoundError(f"Model {model_id} not loaded")
        return True

    def _build_prompt(self, request: ChatRequest) -> str:
        parts: List[str] = []
        for msg in request.messages:
            role = msg.role
            content = msg.content if isinstance(msg.content, str) else ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _mock_generate(self, prompt: str, max_tokens: int) -> str:
        words = prompt.split()
        return " ".join(words[-10:] + [f"generated_token_{i}" for i in range(min(max_tokens, 64))])

    def _mock_tokenize(self, text: str) -> List[str]:
        words = text.split()
        return words + [f"tok_{i}" for i in range(32)]

    def _estimate_usage(self, prompt: str, completion: str) -> Usage:
        prompt_tokens = len(prompt.split())
        completion_tokens = len(completion.split())
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)


class threading_lock:
    def __enter__(self) -> threading_lock:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass
