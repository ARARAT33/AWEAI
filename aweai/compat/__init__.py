"""AWEAI compatibility layer — drop-in local replacements for major AI APIs."""

from __future__ import annotations

from aweai.compat import anthropic, google, local_models, openai, providers, router, types

__all__ = [
    "anthropic",
    "google",
    "local_models",
    "openai",
    "providers",
    "router",
    "types",
]
