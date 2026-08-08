"""Automatic model selection based on detected hardware.

Requirement #8: AWEAI learns the machine's resources and picks the best
model that will actually run there (fastest capable model that fits RAM).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from aweai.models import MODELS
from aweai.hardware import HardwareInfo, detect


def _fit_score(m: Dict, hw: HardwareInfo) -> float:
    """Score how well a model fits the hardware. Higher = better fit."""
    # must fit in RAM (with a little headroom)
    if m["min_ram_gb"] > hw.ram_total_gb * 0.9:
        return -1.0
    if hw.gpu_total_vram_gb and m["vram_gb"] > hw.gpu_total_vram_gb:
        # could still run on CPU+RAM, penalize but don't exclude
        penalty = (m["vram_gb"] - hw.gpu_total_vram_gb) * 0.5
    else:
        penalty = 0.0
    # bigger params = smarter, but more resources
    size_score = m["params_b"] * 10.0
    # GPU is much faster; boost if the model fits in VRAM
    gpu_boost = 50.0 if hw.gpu_total_vram_gb and m["vram_gb"] <= hw.gpu_total_vram_gb else 0.0
    return size_score + gpu_boost - penalty


def pick_best_model(hw: Optional[HardwareInfo] = None, ram_headroom: float = 0.9) -> Optional[Dict]:
    """Pick the best model from the catalog for the detected hardware."""
    hw = hw or detect()
    best = None
    best_score = -1.0
    for m in MODELS:
        score = _fit_score(m, hw)
        if score > best_score:
            best_score = score
            best = m
    return best


def suggest_models(hw: Optional[HardwareInfo] = None, limit: int = 5) -> List[Dict]:
    """Return top-N best fitting models, best first."""
    hw = hw or detect()
    scored = [( _fit_score(m, hw), m) for m in MODELS]
    scored = [(s, m) for s, m in scored if s >= 0.0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


def pick_for_android() -> Dict:
    """Models that run on a phone (tiny + int4)."""
    return get_fallback()


def get_fallback() -> Dict:
    for m in MODELS:
        if m["id"] == "qwen2.5-0.5b":
            return m
    return MODELS[0]
