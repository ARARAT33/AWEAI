"""Pre-trained model catalog for hardware-aware selection.

AWEAI ships **no built-in AI weights** — this catalog describes well-known
open model IDs (sizes, VRAM/RAM footprint) so the factory can *recommend*
which external model would fit the detected hardware. The IDs are used with
`aweai.models.apis.APIManager` (BYOK / OpenAI-compatible endpoints) or with
the local inference shim in `aweai.models.inference`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Catalog entries: id, family, params_b (billions), min_ram_gb, vram_gb, task
MODELS: List[Dict] = [
    {"id": "qwen2.5-0.5b", "family": "Qwen", "params_b": 0.5, "min_ram_gb": 1.0, "vram_gb": 0.5, "task": "text"},
    {"id": "qwen2.5-1.5b", "family": "Qwen", "params_b": 1.5, "min_ram_gb": 2.0, "vram_gb": 1.5, "task": "text"},
    {"id": "llama-3.2-1b", "family": "Llama", "params_b": 1.2, "min_ram_gb": 2.0, "vram_gb": 1.2, "task": "text"},
    {"id": "gemma-2-2b", "family": "Gemma", "params_b": 2.6, "min_ram_gb": 4.0, "vram_gb": 2.6, "task": "text"},
    {"id": "qwen2.5-3b", "family": "Qwen", "params_b": 3.0, "min_ram_gb": 4.0, "vram_gb": 3.0, "task": "text"},
    {"id": "phi-3-mini", "family": "Phi", "params_b": 3.8, "min_ram_gb": 6.0, "vram_gb": 4.0, "task": "text"},
    {"id": "llama-3.2-3b", "family": "Llama", "params_b": 3.2, "min_ram_gb": 6.0, "vram_gb": 3.2, "task": "text"},
    {"id": "mistral-7b", "family": "Mistral", "params_b": 7.0, "min_ram_gb": 8.0, "vram_gb": 7.0, "task": "text"},
    {"id": "llama-3.1-8b", "family": "Llama", "params_b": 8.0, "min_ram_gb": 12.0, "vram_gb": 8.0, "task": "text"},
    {"id": "qwen2.5-7b", "family": "Qwen", "params_b": 7.6, "min_ram_gb": 12.0, "vram_gb": 8.0, "task": "text"},
    {"id": "gemma-2-9b", "family": "Gemma", "params_b": 9.2, "min_ram_gb": 16.0, "vram_gb": 10.0, "task": "text"},
    {"id": "llama-3.1-70b", "family": "Llama", "params_b": 70.0, "min_ram_gb": 48.0, "vram_gb": 40.0, "task": "text"},
]

_FALLBACK_ID = "qwen2.5-0.5b"


def get_model(model_id: str) -> Optional[Dict]:
    """Return catalog entry for a model id (case-insensitive), or None."""
    for m in MODELS:
        if m["id"].lower() == model_id.lower():
            return m
    return None


def list_models() -> List[Dict]:
    """Return the full catalog."""
    return list(MODELS)


def get_fallback() -> Dict:
    """Tiny model that runs anywhere (phones, edge, CI)."""
    return get_model(_FALLBACK_ID) or MODELS[0]


def catalog_stats() -> Dict:
    return {
        "count": len(MODELS),
        "families": sorted({m["family"] for m in MODELS}),
        "smallest": get_fallback()["id"],
    }
