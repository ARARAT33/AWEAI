# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Universal AI ecosystem gateway.

AWEAI is the control plane: AI-company/provider capabilities are normalized
into one registry and one routing policy. This module deliberately does not
pretend that a provider has an API it does not expose; adapters are declared
as capabilities and become executable when credentials/endpoints exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from aweai.cmd.common import APP_DIR

app = typer.Typer(help="AWEAI Universal AI Ecosystem: one control plane for AI providers and tools")

# Broad ecosystem registry. Capability names are intentionally normalized so
# AWEAI can expose one interface even when vendor APIs differ.
PROVIDERS = {
    "openai": {"name": "OpenAI", "key": "OPENAI_API_KEY", "caps": "chat,reasoning,code,embeddings,image,audio,realtime,moderation,files,batch,fine_tune"},
    "anthropic": {"name": "Anthropic", "key": "ANTHROPIC_API_KEY", "caps": "chat,reasoning,code,vision,files,batch"},
    "google": {"name": "Google Gemini", "key": "GOOGLE_API_KEY", "caps": "chat,reasoning,code,vision,audio,video,embeddings,search"},
    "microsoft": {"name": "Microsoft Azure AI", "key": "AZURE_OPENAI_API_KEY", "caps": "chat,code,vision,embeddings,agents,search,speech"},
    "meta": {"name": "Meta AI", "key": "META_API_KEY", "caps": "models,vision,embeddings,research"},
    "xai": {"name": "xAI", "key": "XAI_API_KEY", "caps": "chat,reasoning,code,vision,search"},
    "mistral": {"name": "Mistral AI", "key": "MISTRAL_API_KEY", "caps": "chat,reasoning,code,embeddings,ocr"},
    "cohere": {"name": "Cohere", "key": "COHERE_API_KEY", "caps": "chat,embeddings,rerank,classify"},
    "deepseek": {"name": "DeepSeek", "key": "DEEPSEEK_API_KEY", "caps": "chat,reasoning,code"},
    "qwen": {"name": "Alibaba Qwen", "key": "DASHSCOPE_API_KEY", "caps": "chat,reasoning,code,vision,audio,embeddings"},
    "zhipu": {"name": "Zhipu AI", "key": "ZHIPUAI_API_KEY", "caps": "chat,reasoning,vision,embeddings"},
    "moonshot": {"name": "Moonshot AI", "key": "MOONSHOT_API_KEY", "caps": "chat,reasoning,code,search"},
    "minimax": {"name": "MiniMax", "key": "MINIMAX_API_KEY", "caps": "chat,audio,video,image"},
    "groq": {"name": "Groq", "key": "GROQ_API_KEY", "caps": "chat,reasoning,code,audio"},
    "together": {"name": "Together AI", "key": "TOGETHER_API_KEY", "caps": "chat,code,image,embeddings,fine_tune"},
    "fireworks": {"name": "Fireworks AI", "key": "FIREWORKS_API_KEY", "caps": "chat,code,image,embeddings,fine_tune"},
    "perplexity": {"name": "Perplexity", "key": "PERPLEXITY_API_KEY", "caps": "chat,reasoning,search"},
    "openrouter": {"name": "OpenRouter", "key": "OPENROUTER_API_KEY", "caps": "routing,chat,reasoning,code,vision,embeddings"},
    "huggingface": {"name": "Hugging Face", "key": "HF_TOKEN", "caps": "models,datasets,inference,spaces,training"},
    "replicate": {"name": "Replicate", "key": "REPLICATE_API_TOKEN", "caps": "models,image,audio,video,training"},
    "stability": {"name": "Stability AI", "key": "STABILITY_API_KEY", "caps": "image,video,3d"},
    "elevenlabs": {"name": "ElevenLabs", "key": "ELEVENLABS_API_KEY", "caps": "speech,voice,transcription"},
    "assemblyai": {"name": "AssemblyAI", "key": "ASSEMBLYAI_API_KEY", "caps": "transcription,speech,intelligence"},
    "groqcloud": {"name": "Groq Cloud", "key": "GROQ_API_KEY", "caps": "chat,code,transcription"},
    "ollama": {"name": "Ollama", "key": "OLLAMA_HOST", "caps": "local_models,chat,code,embeddings"},
    "lmstudio": {"name": "LM Studio", "key": "LMSTUDIO_HOST", "caps": "local_models,chat,code,embeddings"},
}

CAPABILITIES = [
    "chat", "reasoning", "code", "vision", "embeddings", "rerank", "search",
    "image", "audio", "speech", "transcription", "video", "3d", "realtime",
    "moderation", "files", "batch", "fine_tune", "training", "datasets", "models",
    "agents", "evaluation", "monitoring", "routing", "local_models", "ocr", "classify",
]


def _policy_file() -> Path:
    return APP_DIR / "ecosystem_policy.json"


def _load_policy() -> dict:
    p = _policy_file()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "mode": "aweai_only",
        "allow_direct_vendor_tools": False,
        "require_gateway": True,
        "require_audit": True,
        "fallback": "aweai_registry",
        "version": 1,
    }


@app.command("catalog")
def catalog(category: Optional[str] = typer.Option(None, "--capability", "-c")):
    """Show the normalized AI-company/provider catalog."""
    rows = []
    for pid, info in PROVIDERS.items():
        caps = info["caps"].split(",")
        if category and category not in caps:
            continue
        rows.append({"id": pid, "name": info["name"], "configured": bool(os.environ.get(info["key"])), "capabilities": caps})
    typer.echo(json.dumps({"count": len(rows), "providers": rows}, indent=2))


@app.command("capabilities")
def capabilities():
    """List every normalized AWEAI capability surface."""
    matrix = {c: [p for p, x in PROVIDERS.items() if c in x["caps"].split(",")] for c in CAPABILITIES}
    typer.echo(json.dumps(matrix, indent=2))


@app.command("policy")
def policy(action: str = typer.Argument("show", help="show|enforce|allow-direct")):
    """Control the AWEAI-only execution policy."""
    data = _load_policy()
    if action == "enforce":
        data.update({"mode": "aweai_only", "allow_direct_vendor_tools": False, "require_gateway": True, "require_audit": True})
        _policy_file().parent.mkdir(parents=True, exist_ok=True)
        _policy_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif action == "allow-direct":
        data.update({"mode": "hybrid", "allow_direct_vendor_tools": True})
        _policy_file().parent.mkdir(parents=True, exist_ok=True)
        _policy_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif action != "show":
        raise typer.BadParameter("use show|enforce|allow-direct")
    typer.echo(json.dumps(data, indent=2))


@app.command("route")
def route(capability: str = typer.Argument(...), preferred: Optional[str] = typer.Option(None, "--preferred", "-p")):
    """Choose an AWEAI-managed provider for a capability."""
    if capability not in CAPABILITIES:
        raise typer.BadParameter(f"unknown capability: {capability}")
    candidates = [p for p, x in PROVIDERS.items() if capability in x["caps"].split(",")]
    if preferred:
        if preferred not in candidates:
            typer.echo(json.dumps({"ok": False, "error": "preferred provider lacks capability", "candidates": candidates}, indent=2))
            raise typer.Exit(code=1)
        candidates = [preferred] + [p for p in candidates if p != preferred]
    configured = [p for p in candidates if os.environ.get(PROVIDERS[p]["key"])]
    typer.echo(json.dumps({"ok": bool(candidates), "capability": capability, "selected": (configured or candidates)[0] if candidates else None, "configured_candidates": configured, "all_candidates": candidates, "gateway": "AWEAI"}, indent=2))


@app.command("audit")
def audit():
    """Report ecosystem coverage and policy readiness without contacting vendors."""
    configured = [p for p, x in PROVIDERS.items() if os.environ.get(x["key"])]
    covered = {c: any(c in PROVIDERS[p]["caps"].split(",") for p in PROVIDERS) for c in CAPABILITIES}
    missing = [c for c, ok in covered.items() if not ok]
    typer.echo(json.dumps({"providers": len(PROVIDERS), "configured_providers": configured, "capabilities": len(CAPABILITIES), "missing_capabilities": missing, "policy": _load_policy(), "gateway": "AWEAI"}, indent=2))


@app.command("contract")
def contract():
    """Print the AWEAI universal tool contract used by future adapters."""
    contract = {
        "name": "AWEAI Universal AI Tool Contract",
        "version": "1.0",
        "principle": "AI applications use AWEAI as the single tool/control plane; vendor-specific APIs are implementation details.",
        "operations": ["discover", "authorize", "route", "execute", "stream", "cancel", "retry", "evaluate", "audit", "cache", "trace"],
        "input": ["capability", "task", "payload", "constraints", "preferred_provider", "budget", "latency"],
        "output": ["provider", "model", "result", "usage", "latency", "trace_id", "warnings", "audit_id"],
        "safety": ["explicit_credentials", "least_privilege", "no_secret_logging", "audit_trail", "provider_fallback", "dry_run"],
    }
    typer.echo(json.dumps(contract, indent=2))
