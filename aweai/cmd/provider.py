# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Provider commands: API keys, external model calling, fine-tuning
external models (OpenAI/Google/Anthropic/HF/...)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import APP_DIR, err, jdump, ok

app = typer.Typer(help="Providers & integrations: API keys, model calling, external fine-tuning")

PROVIDERS = ["openai", "google", "microsoft", "anthropic", "huggingface", "ollama", "local"]
KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "microsoft": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HF_TOKEN",
    "ollama": "OLLAMA_HOST",
}


def _key_file() -> Path:
    return APP_DIR / "keys.json"


def _load_keys() -> Dict[str, str]:
    if _key_file().exists():
        try:
            return json.loads(_key_file().read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@app.command("list")
def list_providers():
    """List supported providers and key status."""
    keys = _load_keys()
    rows = []
    for p in PROVIDERS:
        env = KEY_ENV.get(p)
        has = bool(keys.get(p) or os.environ.get(env or ""))
        rows.append({"provider": p, "key_configured": has, "env_var": env})
    typer.echo(jdump(ok(providers=rows)))


@app.command("set-key")
def set_key(
    provider: str = typer.Argument(..., help="Provider name"),
    key: str = typer.Option(..., "--key", "-k", help="API key"),
):
    """Store an API key locally (aweai keys)."""
    if provider not in KEY_ENV:
        typer.echo(jdump(err(f"unknown provider {provider}; known: {PROVIDERS}")))
        raise typer.Exit(code=1)
    keys = _load_keys()
    keys[provider] = key
    _key_file().parent.mkdir(parents=True, exist_ok=True)
    _key_file().write_text(json.dumps(keys, indent=2), encoding="utf-8")
    os.chmod(_key_file(), 0o600)
    typer.echo(jdump(ok(provider=provider, stored=True, note=f"Also works via env {KEY_ENV[provider]}")))


@app.command("unset-key")
def unset_key(provider: str = typer.Argument(..., help="Provider name")):
    """Remove a stored API key."""
    keys = _load_keys()
    keys.pop(provider, None)
    _key_file().write_text(json.dumps(keys, indent=2), encoding="utf-8")
    typer.echo(jdump(ok(provider=provider, removed=True)))


@app.command("chat")
def chat(
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider"),
    message: str = typer.Option(..., "--message", "-m", help="Chat message"),
    model: Optional[str] = typer.Option(None, "--model", help="Model (optional)"),
):
    """Call an external LLM (BYOK)."""
    try:
        from aweai.integrations import chat as _chat

        typer.echo(jdump(_chat(provider, message, model=model)))
    except Exception as e:
        typer.echo(jdump({"ok": False, "error": str(e),
                          "hint": f"Run: aweai providers set-key {provider} --key YOUR_KEY"}))
        raise typer.Exit(code=1)


@app.command("complete")
def complete(
    prompt: str = typer.Argument(..., help="Prompt"),
    provider: str = typer.Option("openai", "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model"),
):
    """Send a completion prompt to an external model."""
    try:
        from aweai.integrations import complete as _complete

        typer.echo(jdump(_complete(prompt, provider=provider, model=model)))
    except Exception as e:
        typer.echo(jdump({"ok": False, "error": str(e)}))
        raise typer.Exit(code=1)


@app.command("models")
def provider_models(provider: str = typer.Option("openai", "--provider", "-p")):
    """List known model names for a provider."""
    known = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"],
        "google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "microsoft": ["gpt-4o", "gpt-4o-mini", "gpt-35-turbo"],
        "anthropic": ["claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-haiku"],
        "huggingface": ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B-Instruct"],
        "ollama": ["llama3", "mistral", "gemma2", "qwen2.5"],
    }
    typer.echo(jdump(ok(provider=provider, models=known.get(provider, []))))


@app.command("fine-tune")
def fine_tune(
    provider: str = typer.Option("openai", "--provider", "-p"),
    file: str = typer.Option(..., "--file", "-f", help="Training data (JSONL)"),
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="Base model"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Fine-tune job suffix"),
):
    """Fine-tune an external provider model (needs provider API)."""
    try:
        keys = _load_keys()
        key = keys.get(provider) or os.environ.get(KEY_ENV.get(provider, ""))
        if not key:
            typer.echo(jdump(err("no API key configured; run aweai providers set-key")))
            raise typer.Exit(code=1)
        # OpenAI-style upload + create fine-tune job
        if provider in ("openai", "microsoft"):
            import urllib.request

            base = "https://api.openai.com/v1"
            # 1. upload file
            data = Path(file).read_bytes()
            boundary = "AWEAIboundary"
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nfine-tune\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"train.jsonl\"\r\n"
                f"Content-Type: application/json\r\n\r\n".encode() + data +
                f"\r\n--{boundary}--\r\n".encode()
            )
            req = urllib.request.Request(
                f"{base}/files",
                data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                upload = json.loads(r.read())
            file_id = upload["id"]
            job_payload = json.dumps({"training_file": file_id, "model": model,
                                      **({"suffix": suffix} if suffix else {})}).encode()
            req2 = urllib.request.Request(
                f"{base}/fine_tuning/jobs", data=job_payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req2, timeout=120) as r:
                job = json.loads(r.read())
            typer.echo(jdump(ok(provider=provider, file_id=file_id, job=job.get("id"), status=job.get("status"))))
        else:
            typer.echo(jdump(err("fine-tuning via API currently supported for openai/microsoft; others: use provider dashboards")))
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("check")
def check(provider: str = typer.Argument(..., help="Provider")):
    """Check provider API connectivity with a minimal request."""
    try:
        from aweai.integrations import check_provider

        typer.echo(jdump(check_provider(provider)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)
