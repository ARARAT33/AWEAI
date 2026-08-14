# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI Universal AI Ecosystem tools.

Normalizes provider discovery, capability routing, policy and audit into the
existing AWEAI tool registry. These tools are control-plane operations: they
never claim to execute a vendor capability unless an adapter/credential is
actually available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from aweai.cmd.common import APP_DIR
from aweai.tools.registry import tool

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": ("OpenAI", "OPENAI_API_KEY", "chat,reasoning,code,embeddings,image,audio,realtime,moderation,files,batch,fine_tune"),
    "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", "chat,reasoning,code,vision,files,batch"),
    "google": ("Google Gemini", "GOOGLE_API_KEY", "chat,reasoning,code,vision,audio,video,embeddings,search"),
    "microsoft": ("Microsoft Azure AI", "AZURE_OPENAI_API_KEY", "chat,code,vision,embeddings,agents,search,speech"),
    "meta": ("Meta AI", "META_API_KEY", "models,vision,embeddings,research"),
    "xai": ("xAI", "XAI_API_KEY", "chat,reasoning,code,vision,search"),
    "mistral": ("Mistral AI", "MISTRAL_API_KEY", "chat,reasoning,code,embeddings,ocr"),
    "cohere": ("Cohere", "COHERE_API_KEY", "chat,embeddings,rerank,classify"),
    "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", "chat,reasoning,code"),
    "qwen": ("Alibaba Qwen", "DASHSCOPE_API_KEY", "chat,reasoning,code,vision,audio,embeddings"),
    "zhipu": ("Zhipu AI", "ZHIPUAI_API_KEY", "chat,reasoning,vision,embeddings"),
    "moonshot": ("Moonshot AI", "MOONSHOT_API_KEY", "chat,reasoning,code,search"),
    "minimax": ("MiniMax", "MINIMAX_API_KEY", "chat,audio,video,image"),
    "groq": ("Groq", "GROQ_API_KEY", "chat,reasoning,code,audio"),
    "together": ("Together AI", "TOGETHER_API_KEY", "chat,code,image,embeddings,fine_tune"),
    "fireworks": ("Fireworks AI", "FIREWORKS_API_KEY", "chat,code,image,embeddings,fine_tune"),
    "perplexity": ("Perplexity", "PERPLEXITY_API_KEY", "chat,reasoning,search"),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", "routing,chat,reasoning,code,vision,embeddings"),
    "huggingface": ("Hugging Face", "HF_TOKEN", "models,datasets,inference,spaces,training"),
    "replicate": ("Replicate", "REPLICATE_API_TOKEN", "models,image,audio,video,training"),
    "stability": ("Stability AI", "STABILITY_API_KEY", "image,video,3d"),
    "elevenlabs": ("ElevenLabs", "ELEVENLABS_API_KEY", "speech,voice,transcription"),
    "assemblyai": ("AssemblyAI", "ASSEMBLYAI_API_KEY", "transcription,speech,intelligence"),
    "ollama": ("Ollama", "OLLAMA_HOST", "local_models,chat,code,embeddings"),
    "lmstudio": ("LM Studio", "LMSTUDIO_HOST", "local_models,chat,code,embeddings"),
}
CAPABILITIES = sorted({c for _, _, caps in PROVIDERS.values() for c in caps.split(",")} | {
    "evaluation", "monitoring", "routing", "moderation", "agents", "batch", "fine_tune"
})


def _policy_path() -> Path:
    return APP_DIR / "ecosystem_policy.json"


def _policy() -> dict:
    p = _policy_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"mode": "aweai_only", "allow_direct_vendor_tools": False, "require_gateway": True, "require_audit": True, "version": 1}


@tool("ecosystem_catalog", "ecosystem", "List normalized AI providers, capabilities and credential status", {"capability": "optional string"})
def ecosystem_catalog(capability: Optional[str] = None) -> Dict[str, Any]:
    rows = []
    for pid, (name, key, caps) in PROVIDERS.items():
        values = caps.split(",")
        if capability and capability not in values:
            continue
        rows.append({"id": pid, "name": name, "configured": bool(os.environ.get(key)), "credential_env": key, "capabilities": values})
    return {"provider_count": len(rows), "providers": rows}


@tool("ecosystem_capabilities", "ecosystem", "Build the provider-to-capability matrix used by AWEAI routing")
def ecosystem_capabilities() -> Dict[str, Any]:
    return {c: [p for p, (_, _, caps) in PROVIDERS.items() if c in caps.split(",")] for c in CAPABILITIES}


@tool("ecosystem_route", "ecosystem", "Select an AWEAI-managed provider for a normalized capability", {"capability": "required", "preferred": "optional"})
def ecosystem_route(capability: str, preferred: Optional[str] = None) -> Dict[str, Any]:
    candidates = [p for p, (_, _, caps) in PROVIDERS.items() if capability in caps.split(",")]
    if not candidates:
        return {"ok": False, "capability": capability, "error": "no registered provider capability"}
    if preferred and preferred in candidates:
        candidates = [preferred] + [p for p in candidates if p != preferred]
    configured = [p for p in candidates if os.environ.get(PROVIDERS[p][1])]
    return {"ok": True, "gateway": "AWEAI", "capability": capability, "selected": (configured or candidates)[0], "configured_candidates": configured, "all_candidates": candidates}


@tool("ecosystem_policy", "ecosystem", "Read or enforce the AWEAI-only tool gateway policy", {"action": "show|enforce|hybrid"})
def ecosystem_policy(action: str = "show") -> Dict[str, Any]:
    data = _policy()
    if action in ("enforce", "aweai_only"):
        data.update({"mode": "aweai_only", "allow_direct_vendor_tools": False, "require_gateway": True, "require_audit": True})
        _policy_path().parent.mkdir(parents=True, exist_ok=True)
        _policy_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif action in ("hybrid", "allow_direct"):
        data.update({"mode": "hybrid", "allow_direct_vendor_tools": True})
        _policy_path().parent.mkdir(parents=True, exist_ok=True)
        _policy_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif action != "show":
        return {"ok": False, "error": "action must be show|enforce|hybrid"}
    return data


@tool("ecosystem_audit", "ecosystem", "Audit provider coverage and AWEAI gateway readiness")
def ecosystem_audit() -> Dict[str, Any]:
    configured = [p for p, (_, key, _) in PROVIDERS.items() if os.environ.get(key)]
    matrix = ecosystem_capabilities()
    return {"providers": len(PROVIDERS), "configured_providers": configured, "capabilities": len(CAPABILITIES), "uncovered_capabilities": [c for c in CAPABILITIES if not matrix.get(c)], "policy": _policy(), "gateway": "AWEAI"}


@tool("ecosystem_contract", "ecosystem", "Return the universal AWEAI provider adapter contract")
def ecosystem_contract() -> Dict[str, Any]:
    return {
        "name": "AWEAI Universal AI Tool Contract", "version": "1.0",
        "operations": ["discover", "authorize", "route", "execute", "stream", "cancel", "retry", "evaluate", "audit", "cache", "trace"],
        "input": ["capability", "task", "payload", "constraints", "preferred_provider", "budget", "latency"],
        "output": ["provider", "model", "result", "usage", "latency", "trace_id", "warnings", "audit_id"],
        "safety": ["explicit_credentials", "least_privilege", "no_secret_logging", "audit_trail", "fallback", "dry_run"],
    }
