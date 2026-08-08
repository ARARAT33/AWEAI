"""LLM client abstraction for AWEAI.

A thin protocol + two concrete implementations:

* ``EchoClient`` — returns the prompt back verbatim (great for tests and
  offline demos; no network required).
* ``OpenAICompatClient`` — talks to any OpenAI-compatible ``/chat/completions``
  endpoint (OpenAI, Groq, Together, local vLLM/Ollama servers, etc.).

Both implement the same async ``complete``/``stream`` interface so an
``Agent`` can swap backends without any code changes.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional

import httpx


class LLMClient(ABC):
    """Abstract LLM client interface used by :class:`aweai.agent.Agent`."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the assistant reply for ``messages``."""

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield assistant reply chunks. Default: single chunk."""
        text = await self.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        yield text


class EchoClient(LLMClient):
    """Offline client that echoes the last user message.

    Useful for demos, tests, and CI where no API key is available.
    """

    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Return the most recent user message content.
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""


class OpenAICompatClient(LLMClient):
    """Client for OpenAI-compatible chat completion endpoints.

    Args:
        api_key: Provider API key.
        model: Model identifier.
        base_url: API base URL; defaults to OpenAI.
        temperature: Default sampling temperature.
        max_tokens: Default max tokens.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        if api_key is None and base_url is None:
            raise ValueError(
                "OpenAICompatClient requires an api_key (or a local base_url)."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from LLM: {json.dumps(data)[:300]}"
            ) from exc

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta:
                    yield delta


def make_client(config) -> LLMClient:
    """Build an appropriate client from an :class:`AWEConfig`.

    Returns an ``EchoClient`` when no ``api_key`` is configured, which
    makes the toolkit usable out of the box.
    """
    if not config.api_key and not config.base_url:
        return EchoClient()
    return OpenAICompatClient(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )
