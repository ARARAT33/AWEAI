"""API manager: connect and manage cloud LLM APIs (BYOK).

Supported providers (OpenAI-compatible chat completions):
    openai, anthropic (via OpenAI-compat endpoint), gemini, groq, together,
    mistral, deepseek, ollama (local), lmstudio, custom.

Keys are stored in ~/.aweai/api_keys.json (0600).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from aweai.config import get_config, get_api_keys

DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mistral": "mistral-small-latest",
    "deepseek": "deepseek-chat",
    "ollama": "llama3.2",
    "lmstudio": "local-model",
}


class APIManager:
    """Minimal OpenAI-compatible chat client with zero deps."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.cfg = get_config()
        self.provider = provider or self.cfg.get("api_provider", "openai")
        self.base_url = base_url or self.cfg.get("api_base_url") or DEFAULT_ENDPOINTS.get(self.provider, DEFAULT_ENDPOINTS["openai"])
        self.model = model or self.cfg.get("api_model") or DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.api_key = api_key
        if not self.api_key and self.provider != "ollama":
            self.api_key = get_api_keys().get(self.provider)

    def set_key(self, key: str) -> None:
        self.api_key = key
        get_api_keys().set(self.provider, key)

    def _url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            self._url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:400]
            raise RuntimeError(f"API error {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"API request failed: {e}") from e

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, **kwargs)

    def info(self) -> Dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "has_key": bool(self.api_key),
        }


def providers() -> Dict[str, str]:
    return dict(DEFAULT_ENDPOINTS)


def check_provider(provider: str) -> Dict:
    """Run a tiny smoke check against a provider config."""
    mgr = APIManager(provider=provider)
    return mgr.info()
