"""Provider adapters with BYOK support, response normalization, and retry logic."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

from aweai.compat.types import (
    ChatMessage,
    ChatResponse,
    CompletionChoice,
    EmbeddingResponse,
    FunctionCall,
    ModelInfo,
    ModelsResponse,
    Usage,
    APIError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ServerError,
    ValidationError,
)


class BaseProvider:
    def __init__(self, name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.name = name
        self.api_key = api_key or os.environ.get(f"{name.upper()}_API_KEY", "")
        self.base_url = base_url or self._default_url()
        self.model = model or self._default_model()
        self.timeout = 120
        self.max_retries = 3
        self.retry_delay = 1.0

    def _default_url(self) -> str:
        raise NotImplementedError

    def _default_model(self) -> str:
        raise NotImplementedError

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        req_body = json.dumps(body or {}).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=req_body, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body_text = e.read().decode("utf-8", errors="ignore")[:500]
                if e.code == 401:
                    raise AuthenticationError(f"{self.name}: {body_text}")
                if e.code == 404:
                    raise ModelNotFoundError(f"{self.name}: {body_text}")
                if e.code == 429:
                    retry_after = self._parse_retry_after(e.headers.get("Retry-After"))
                    raise RateLimitError(f"{self.name}: {body_text}", retry_after=retry_after)
                if 500 <= e.code < 600:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise ServerError(f"{self.name}: {body_text}")
                raise RuntimeError(f"{self.name} HTTP {e.code}: {body_text}") from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise ServerError(f"{self.name}: network error") from e
        raise ServerError(f"{self.name}: max retries exceeded")

    def _parse_retry_after(self, header: Optional[str]) -> int:
        try:
            return int(header)
        except (TypeError, ValueError):
            return 60

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def embeddings(self, texts: List[str], model: Optional[str] = None) -> EmbeddingResponse:
        raise NotImplementedError

    def models(self) -> ModelsResponse:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__("openai", api_key, base_url, model)

    def _default_url(self) -> str:
        return "https://api.openai.com/v1"

    def _default_model(self) -> str:
        return "gpt-4o-mini"

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        data = self._request("POST", "/chat/completions", payload)
        return self._parse_chat(data)

    def embeddings(self, texts: List[str], model: Optional[str] = None) -> EmbeddingResponse:
        payload = {"model": model or self.model + "-embedding", "input": texts}
        data = self._request("POST", "/embeddings", payload)
        return self._parse_embeddings(data)

    def models(self) -> ModelsResponse:
        data = self._request("GET", "/models")
        model_list = [ModelInfo(id=m["id"], owned_by=m.get("owned_by", "openai")) for m in data.get("data", [])]
        return ModelsResponse(data=model_list)

    def _parse_chat(self, data: Dict[str, Any]) -> ChatResponse:
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            fc = None
            if msg.get("function_call"):
                fc = FunctionCall(msg["function_call"]["name"], msg["function_call"]["arguments"])
            choice = CompletionChoice(
                index=c.get("index", 0),
                text=msg.get("content", ""),
                finish_reason=c.get("finish_reason", "stop"),
                function_call=fc,
                tool_calls=msg.get("tool_calls"),
            )
            choices.append(choice)
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"], total_tokens=u["total_tokens"])
        return ChatResponse(id=data.get("id", ""), created=data.get("created", 0), model=data.get("model", self.model), choices=choices, usage=usage)

    def _parse_embeddings(self, data: Dict[str, Any]) -> EmbeddingResponse:
        emb_data = []
        for item in data.get("data", []):
            emb_data.append({"object": item.get("object", "embedding"), "embedding": item.get("embedding", []), "index": item.get("index", 0)})
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"], total_tokens=u["total_tokens"])
        return EmbeddingResponse(data=emb_data, model=data.get("model", self.model), usage=usage)


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__("google", api_key, base_url, model)

    def _default_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai"

    def _default_model(self) -> str:
        return "gemini-2.0-flash"

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        data = self._request("POST", "/chat/completions", payload)
        return self._parse_chat(data)

    def embeddings(self, texts: List[str], model: Optional[str] = None) -> EmbeddingResponse:
        payload = {"model": model or self.model, "content": {"parts": [{"text": t} for t in texts]}}
        data = self._request("POST", f"/models/{model or self.model}:embedText", payload)
        return self._parse_embeddings(data)

    def models(self) -> ModelsResponse:
        data = self._request("GET", "/models")
        model_list = [ModelInfo(id=m["name"], owned_by="google") for m in data.get("models", [])]
        return ModelsResponse(data=model_list)

    def _parse_chat(self, data: Dict[str, Any]) -> ChatResponse:
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choice = CompletionChoice(
                index=c.get("index", 0),
                text=msg.get("content", ""),
                finish_reason=c.get("finish_reason", "stop"),
            )
            choices.append(choice)
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0), total_tokens=u.get("total_tokens", 0))
        return ChatResponse(id=data.get("id", ""), created=data.get("created", 0), model=data.get("model", self.model), choices=choices, usage=usage)

    def _parse_embeddings(self, data: Dict[str, Any]) -> EmbeddingResponse:
        return EmbeddingResponse(data=[], model=self.model)


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__("anthropic", api_key, base_url, model)

    def _default_url(self) -> str:
        return "https://api.anthropic.com/v1"

    def _default_model(self) -> str:
        return "claude-3-5-haiku-latest"

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.7, system: Optional[str] = None, **kwargs: Any) -> ChatResponse:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if system:
            payload["system"] = system
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        data = self._request("POST", "/messages", payload, headers={"anthropic-version": "2023-06-01"})
        return self._parse_chat(data)

    def models(self) -> ModelsResponse:
        data = self._request("GET", "/models")
        model_list = [ModelInfo(id=m.get("id", "unknown"), owned_by="anthropic") for m in data.get("data", [])]
        return ModelsResponse(data=model_list)

    def _parse_chat(self, data: Dict[str, Any]) -> ChatResponse:
        choices = []
        content = data.get("content", [])
        text = ""
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    text += block.get("text", "")
        elif isinstance(content, str):
            text = content
        choice = CompletionChoice(index=0, text=text, finish_reason=data.get("stop_reason", "stop"))
        choices.append(choice)
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(prompt_tokens=u.get("input_tokens", 0), completion_tokens=u.get("output_tokens", 0), total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0))
        return ChatResponse(id=data.get("id", ""), model=data.get("model", self.model), choices=choices, usage=usage)


class HuggingFaceProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__("huggingface", api_key, base_url, model)

    def _default_url(self) -> str:
        return "https://api-inference.huggingface.co"

    def _default_model(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.2"

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        prompt = self._build_prompt(messages)
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "return_full_text": False}}
        data = self._request("POST", f"/models/{self.model}", payload)
        if isinstance(data, list):
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text", "")
        else:
            text = str(data)
        choice = CompletionChoice(index=0, text=text, finish_reason="stop")
        return ChatResponse(model=self.model, choices=[choice])

    def embeddings(self, texts: List[str], model: Optional[str] = None) -> EmbeddingResponse:
        payload = {"inputs": texts}
        data = self._request("POST", f"/models/{model or 'sentence-transformers/all-MiniLM-L6-v2'}", payload)
        emb_data = []
        if isinstance(data, list):
            for i, emb in enumerate(data):
                emb_data.append({"object": "embedding", "embedding": emb, "index": i})
        return EmbeddingResponse(data=emb_data, model=model or "sentence-transformers/all-MiniLM-L6-v2")

    def models(self) -> ModelsResponse:
        return ModelsResponse(data=[ModelInfo(id=self.model, owned_by="huggingface")])

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<{role}>: {content}")
        parts.append("<assistant>:")
        return "\n".join(parts)


class OllamaProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__("ollama", api_key, base_url, model)
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    def _default_url(self) -> str:
        return "http://localhost:11434/v1"

    def _default_model(self) -> str:
        return "llama3.2"

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": {"num_predict": max_tokens, "temperature": temperature}}
        data = self._request("POST", "/chat/completions", payload)
        return self._parse_chat(data)

    def models(self) -> ModelsResponse:
        data = self._request("GET", "/models")
        model_list = [ModelInfo(id=m.get("name", "unknown"), owned_by="ollama") for m in data.get("models", [])]
        return ModelsResponse(data=model_list)

    def _parse_chat(self, data: Dict[str, Any]) -> ChatResponse:
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choice = CompletionChoice(index=c.get("index", 0), text=msg.get("content", ""), finish_reason=c.get("finish_reason", "stop"))
            choices.append(choice)
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0), total_tokens=u.get("total_tokens", 0))
        return ChatResponse(id=data.get("id", ""), model=data.get("model", self.model), choices=choices, usage=usage)


PROVIDERS: Dict[str, type] = {
    "openai": OpenAIProvider,
    "google": GoogleProvider,
    "gemini": GoogleProvider,
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,
    "ollama": OllamaProvider,
    "lmstudio": OllamaProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    cls = PROVIDERS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return cls(**kwargs)


def normalize_response(data: Dict[str, Any], provider_name: str) -> ChatResponse:
    if provider_name in ("openai", "ollama", "lmstudio"):
        provider = OpenAIProvider()
        return provider._parse_chat(data)
    if provider_name in ("google", "gemini"):
        provider = GoogleProvider()
        return provider._parse_chat(data)
    if provider_name in ("anthropic", "claude"):
        provider = AnthropicProvider()
        return provider._parse_chat(data)
    raise ValueError(f"Cannot normalize response for provider: {provider_name}")
