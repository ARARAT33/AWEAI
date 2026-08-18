"""AI-tool integrations (v3.0).

Adapters for the world's major AI platforms — OpenAI, Google, Microsoft,
Anthropic, HuggingFace — plus a generic OpenAI-compatible client.

All adapters are **BYOK (bring-your-own-key)**: credentials come from the
environment (``OPENAI_API_KEY``, ``GOOGLE_API_KEY``, ``AZURE_OPENAI_KEY``,
``ANTHROPIC_API_KEY``, ``HF_TOKEN``) or from ``~/.aweai/config.json``.
No keys are bundled; calls without keys return a helpful diagnostic so the
factory remains fully functional offline.

The package also exposes a ``list_tools()`` registry so the terminal,
autotest and UI can enumerate what is available.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from aweai.config import get_config


def _get_key(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    cfg = get_config()
    return cfg.get(name.lower(), None)


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
        "auth": "Bearer",
    },
    "google": {
        "name": "Google Gemini",
        "env_key": "GOOGLE_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "default_model": "gemini-1.5-flash",
        "auth": "key",
    },
    "microsoft": {
        "name": "Microsoft Azure OpenAI",
        "env_key": "AZURE_OPENAI_KEY",
        "url": "https://{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-01",
        "default_model": "gpt-4o-mini",
        "auth": "api-key",
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-3-5-haiku-latest",
        "auth": "x-api-key",
    },
    "huggingface": {
        "name": "Hugging Face",
        "env_key": "HF_TOKEN",
        "url": "https://api-inference.huggingface.co/models/{model}",
        "default_model": "gpt2",
        "auth": "Bearer",
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "url": "https://api.deepseek.com/chat/completions",
        "default_model": "deepseek-chat",
        "auth": "Bearer",
    },
    "qwen": {
        "name": "Alibaba Qwen",
        "env_key": "DASHSCOPE_API_KEY",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "default_model": "qwen-max",
        "auth": "Bearer",
    },
    "zhipu": {
        "name": "Zhipu AI",
        "env_key": "ZHIPUAI_API_KEY",
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "default_model": "glm-4",
        "auth": "Bearer",
    },
    "moonshot": {
        "name": "Moonshot AI",
        "env_key": "MOONSHOT_API_KEY",
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "default_model": "moonshot-v1-8k",
        "auth": "Bearer",
    },
    "groq": {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.3-70b-versatile",
        "auth": "Bearer",
    },
    "together": {
        "name": "Together AI",
        "env_key": "TOGETHER_API_KEY",
        "url": "https://api.together.xyz/v1/chat/completions",
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "auth": "Bearer",
    },
    "fireworks": {
        "name": "Fireworks AI",
        "env_key": "FIREWORKS_API_KEY",
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "default_model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "auth": "Bearer",
    },
    "perplexity": {
        "name": "Perplexity",
        "env_key": "PERPLEXITY_API_KEY",
        "url": "https://api.perplexity.ai/chat/completions",
        "default_model": "sonar",
        "auth": "Bearer",
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "auto",
        "auth": "Bearer",
    },
    "xai": {
        "name": "xAI",
        "env_key": "XAI_API_KEY",
        "url": "https://api.x.ai/v1/chat/completions",
        "default_model": "grok-beta",
        "auth": "Bearer",
    },
    "mistral": {
        "name": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "default_model": "mistral-small-latest",
        "auth": "Bearer",
    },
    "cohere": {
        "name": "Cohere",
        "env_key": "COHERE_API_KEY",
        "url": "https://api.cohere.com/v2/chat",
        "default_model": "command-r-plus",
        "auth": "Bearer",
    },
    "ollama": {
        "name": "Ollama",
        "env_key": "OLLAMA_HOST",
        "url": "http://localhost:11434/api/chat",
        "default_model": "llama3",
        "auth": "none",
    },
    "lmstudio": {
        "name": "LM Studio",
        "env_key": "LMSTUDIO_HOST",
        "url": "http://localhost:1234/v1/chat/completions",
        "default_model": "local-model",
        "auth": "none",
    },
    "meta": {
        "name": "Meta AI",
        "env_key": "META_API_KEY",
        "url": "https://api.meta.ai/v1/chat/completions",
        "default_model": "llama-3.1",
        "auth": "Bearer",
    },
    "minimax": {
        "name": "MiniMax",
        "env_key": "MINIMAX_API_KEY",
        "url": "https://api.minimax.chat/v1/text/chatcompletion_v2",
        "default_model": "abab6.5g-chat",
        "auth": "Bearer",
    },
    "replicate": {
        "name": "Replicate",
        "env_key": "REPLICATE_API_TOKEN",
        "url": "https://api.replicate.com/v1/predictions",
        "default_model": "meta/llama-2-70b-chat",
        "auth": "Token",
    },
    "stability": {
        "name": "Stability AI",
        "env_key": "STABILITY_API_KEY",
        "url": "https://api.stability.ai/v1/generation",
        "default_model": "stable-diffusion-v1-6",
        "auth": "Bearer",
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "env_key": "ELEVENLABS_API_KEY",
        "url": "https://api.elevenlabs.io/v1/text-to-speech",
        "default_model": "eleven_monolingual_v1",
        "auth": "xi-api-key",
    },
    "assemblyai": {
        "name": "AssemblyAI",
        "env_key": "ASSEMBLYAI_API_KEY",
        "url": "https://api.assemblyai.com/v2/transcript",
        "default_model": "default",
        "auth": "authorization",
    },
}


def list_providers() -> List[Dict[str, Any]]:
    out = []
    for pid, info in PROVIDERS.items():
        out.append({
            "id": pid,
            "name": info["name"],
            "configured": bool(_get_key(info["env_key"])),
        })
    return out


def list_tools() -> Dict[str, Any]:
    """Registry used by terminal/UI/autotest."""
    return {
        "providers": list_providers(),
        "models": list(PROVIDERS.keys()),
        "actions": ["chat", "embed", "generate_image", "complete"],
    }


def _chat_openai(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    model = model or info["default_model"]
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": message}], "max_tokens": 500}
    data = _post_json(info["url"], headers, payload)
    try:
        return {"ok": True, "provider": provider, "model": model,
                "reply": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_google(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    model = model or info["default_model"]
    url = info["url"].format(model=model)
    payload = {"contents": [{"parts": [{"text": message}]}]}
    data = _post_json(f"{url}?key={key}", {}, payload)
    try:
        return {"ok": True, "provider": provider, "model": model,
                "reply": data["candidates"][0]["content"]["parts"][0]["text"]}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_azure(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "YOUR_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model or info["default_model"])
    url = info["url"].format(endpoint=endpoint, deployment=deployment)
    headers = {"api-key": key}
    payload = {"messages": [{"role": "user", "content": message}], "max_tokens": 500}
    data = _post_json(url, headers, payload)
    try:
        return {"ok": True, "provider": provider, "model": deployment,
                "reply": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_anthropic(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    model = model or info["default_model"]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    payload = {"model": model, "max_tokens": 500,
               "messages": [{"role": "user", "content": message}]}
    data = _post_json(info["url"], headers, payload)
    try:
        return {"ok": True, "provider": provider, "model": model,
                "reply": data["content"][0]["text"]}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_huggingface(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    model = model or info["default_model"]
    url = info["url"].format(model=model)
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"inputs": message}
    data = _post_json(url, headers, payload)
    try:
        if isinstance(data, list) and data:
            return {"ok": True, "provider": provider, "model": model,
                    "reply": data[0].get("generated_text", str(data[0]))}
        return {"ok": True, "provider": provider, "model": model, "reply": str(data)}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_openai_compatible(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if info["auth"] != "none" and not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    model = model or info["default_model"]
    headers = {}
    if key and info["auth"] == "Bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif key and info["auth"] == "Token":
        headers["Authorization"] = f"Token {key}"
    elif key and info["auth"] == "authorization":
        headers["authorization"] = key
    elif key and info["auth"] == "xi-api-key":
        headers["xi-api-key"] = key
    payload = {"model": model, "messages": [{"role": "user", "content": message}], "max_tokens": 500}
    data = _post_json(info["url"], headers, payload)
    try:
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or data["choices"][0].get("text", "")
            return {"ok": True, "provider": provider, "model": model, "reply": content}
        return {"ok": True, "provider": provider, "model": model, "reply": str(data)}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_cohere(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    key = _get_key(info["env_key"])
    if not key:
        return {"ok": False, "error": f"{info['name']} not configured. Set {info['env_key']}."}
    model = model or info["default_model"]
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": {"text": message}}]}
    data = _post_json(info["url"], headers, payload)
    try:
        reply = data.get("message", {}).get("content", [{}])[0].get("text", str(data))
        return {"ok": True, "provider": provider, "model": model, "reply": reply}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def _chat_ollama(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    info = PROVIDERS[provider]
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/chat"
    model = model or info["default_model"]
    payload = {"model": model, "messages": [{"role": "user", "content": message}], "stream": False}
    data = _post_json(url, {}, payload)
    try:
        return {"ok": True, "provider": provider, "model": model, "reply": data["message"]["content"]}
    except Exception as e:
        return {"ok": False, "error": f"Bad response: {e}", "raw": data}


def chat(provider: str, message: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Send a chat message to a provider (BYOK)."""
    provider = provider.lower()
    if provider not in PROVIDERS:
        return {"ok": False, "error": f"Unknown provider: {provider}. Known: {list(PROVIDERS.keys())}"}
    fn_map = {
        "openai": _chat_openai,
        "google": _chat_google,
        "microsoft": _chat_azure,
        "anthropic": _chat_anthropic,
        "huggingface": _chat_huggingface,
        "cohere": _chat_cohere,
        "ollama": _chat_ollama,
    }
    fn = fn_map.get(provider, _chat_openai_compatible)
    try:
        return fn(provider, message, model)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def complete(prompt: str, provider: str = "openai", model: Optional[str] = None) -> Dict[str, Any]:
    """Alias for chat (completion-style)."""
    return chat(provider, prompt, model)
