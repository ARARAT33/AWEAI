"""Model catalog: SLM/LLM registry covering the whole AI model landscape.

Each entry: id, family, params (B), context (tokens), min_ram_gb,
recommended_vram_gb, quantizations, license, source (HF id), languages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Tiers: tiny (<0.5B), small (0.5-3B), medium (3-13B), large (13-70B), huge (>70B)
MODELS: List[Dict] = [
    # --- Tiny / edge ---
    {
        "id": "qwen2.5-0.5b",
        "family": "Qwen",
        "params_b": 0.5,
        "context": 32768,
        "min_ram_gb": 1.0,
        "vram_gb": 1.0,
        "quantizations": ["int8", "int4"],
        "license": "Apache-2.0",
        "hf": "Qwen/Qwen2.5-0.5B-Instruct",
        "languages": ["en", "zh", "multilingual"],
        "use": "edge, mobile, quick tasks",
        "tier": "tiny",
    },
    {
        "id": "gemma-2-2b",
        "family": "Gemma",
        "params_b": 2.0,
        "context": 8192,
        "min_ram_gb": 2.0,
        "vram_gb": 2.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Gemma",
        "hf": "google/gemma-2-2b-it",
        "languages": ["en", "multilingual"],
        "use": "mobile, laptop CPU",
        "tier": "small",
    },
    {
        "id": "llama-3.2-1b",
        "family": "Llama",
        "params_b": 1.2,
        "context": 131072,
        "min_ram_gb": 1.5,
        "vram_gb": 1.5,
        "quantizations": ["int8", "int4"],
        "license": "Llama-3.2",
        "hf": "meta-llama/Llama-3.2-1B-Instruct",
        "languages": ["en", "multilingual"],
        "use": "mobile, edge",
        "tier": "tiny",
    },
    {
        "id": "phi-3-mini",
        "family": "Phi",
        "params_b": 3.8,
        "context": 131072,
        "min_ram_gb": 4.0,
        "vram_gb": 4.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "MIT",
        "hf": "microsoft/Phi-3-mini-4k-instruct",
        "languages": ["en", "multilingual"],
        "use": "laptop, small GPU",
        "tier": "small",
    },
    {
        "id": "mistral-7b",
        "family": "Mistral",
        "params_b": 7.0,
        "context": 32768,
        "min_ram_gb": 8.0,
        "vram_gb": 8.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Apache-2.0",
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
        "languages": ["en", "fr", "de", "es", "it", "multilingual"],
        "use": "desktop, single GPU",
        "tier": "medium",
    },
    {
        "id": "qwen2.5-7b",
        "family": "Qwen",
        "params_b": 7.6,
        "context": 131072,
        "min_ram_gb": 8.0,
        "vram_gb": 8.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Apache-2.0",
        "hf": "Qwen/Qwen2.5-7B-Instruct",
        "languages": ["en", "zh", "multilingual"],
        "use": "desktop, coding",
        "tier": "medium",
    },
    {
        "id": "llama-3.1-8b",
        "family": "Llama",
        "params_b": 8.0,
        "context": 131072,
        "min_ram_gb": 8.0,
        "vram_gb": 8.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Llama-3.1",
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "languages": ["en", "multilingual"],
        "use": "desktop, coding",
        "tier": "medium",
    },
    {
        "id": "gemma-2-9b",
        "family": "Gemma",
        "params_b": 9.0,
        "context": 8192,
        "min_ram_gb": 12.0,
        "vram_gb": 10.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Gemma",
        "hf": "google/gemma-2-9b-it",
        "languages": ["en", "multilingual"],
        "use": "workstation",
        "tier": "medium",
    },
    {
        "id": "qwen2.5-14b",
        "family": "Qwen",
        "params_b": 14.7,
        "context": 131072,
        "min_ram_gb": 16.0,
        "vram_gb": 16.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Apache-2.0",
        "hf": "Qwen/Qwen2.5-14B-Instruct",
        "languages": ["en", "zh", "multilingual"],
        "use": "workstation, server",
        "tier": "large",
    },
    {
        "id": "llama-3.1-70b",
        "family": "Llama",
        "params_b": 70.0,
        "context": 131072,
        "min_ram_gb": 64.0,
        "vram_gb": 48.0,
        "quantizations": ["int8", "int4"],
        "license": "Llama-3.1",
        "hf": "meta-llama/Llama-3.1-70B-Instruct",
        "languages": ["en", "multilingual"],
        "use": "server, multi-GPU",
        "tier": "huge",
    },
    {
        "id": "qwen2.5-72b",
        "family": "Qwen",
        "params_b": 72.0,
        "context": 131072,
        "min_ram_gb": 64.0,
        "vram_gb": 48.0,
        "quantizations": ["int8", "int4"],
        "license": "Apache-2.0",
        "hf": "Qwen/Qwen2.5-72B-Instruct",
        "languages": ["en", "zh", "multilingual"],
        "use": "server, multi-GPU",
        "tier": "huge",
    },
    {
        "id": "deepseek-r1-distill-7b",
        "family": "DeepSeek",
        "params_b": 7.0,
        "context": 65536,
        "min_ram_gb": 8.0,
        "vram_gb": 8.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "MIT",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "languages": ["en", "zh", "multilingual"],
        "use": "reasoning, desktop",
        "tier": "medium",
    },
    {
        "id": "gemma-3-27b",
        "family": "Gemma",
        "params_b": 27.0,
        "context": 131072,
        "min_ram_gb": 24.0,
        "vram_gb": 24.0,
        "quantizations": ["int8", "int4"],
        "license": "Gemma",
        "hf": "google/gemma-3-27b-it",
        "languages": ["en", "multilingual"],
        "use": "workstation, server",
        "tier": "large",
    },
    {
        "id": "gpt-oss-20b",
        "family": "GPT-OSS",
        "params_b": 20.0,
        "context": 131072,
        "min_ram_gb": 16.0,
        "vram_gb": 16.0,
        "quantizations": ["int8", "int4", "fp16"],
        "license": "Apache-2.0",
        "hf": "openai/gpt-oss-20b",
        "languages": ["en", "multilingual"],
        "use": "workstation, server",
        "tier": "large",
    },
]

# Models usable as a light "always-works" fallback (pure Python, no torch)
FALLBACK_MODEL_ID = "qwen2.5-0.5b"


def get_model(model_id: str) -> Optional[Dict]:
    for m in MODELS:
        if m["id"] == model_id:
            return m
    return None


def list_models(family: Optional[str] = None, tier: Optional[str] = None) -> List[Dict]:
    out = []
    for m in MODELS:
        if family and m["family"].lower() != family.lower():
            continue
        if tier and m.get("tier") != tier:
            continue
        out.append(m)
    return out


def families() -> List[str]:
    fams = []
    for m in MODELS:
        if m["family"] not in fams:
            fams.append(m["family"])
    return fams


def tiers() -> List[str]:
    return ["tiny", "small", "medium", "large", "huge"]


def models_requiring_less_than(ram_gb: float) -> List[Dict]:
    return [m for m in MODELS if m["min_ram_gb"] <= ram_gb]
