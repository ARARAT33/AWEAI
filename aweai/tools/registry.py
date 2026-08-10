"""AWEAI tools registry — the heart of the extension toolkit.

Every tool in the toolkit is a small function with a unique purpose.
Tools are declared with a simple decorator that registers:

* name          — unique machine name (snake_case)
* category      — one of the tool families (security, devops, ...)
* purpose       — one-line description of the tool's UNIQUE purpose
* params        — JSON schema-ish hint dict for UI/CLI rendering
* run           — the callable itself (fn(**params) -> dict/str)

The registry is used by:
* the CLI  (``aweai tool <name> --param value``)
* the CLI  (``aweai tools list`` and ``aweai tools run``)
* the menus (allc/autoallc catalogs, so new tools appear in the
  10,000+ command space automatically)
"""
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional

TOOLS: Dict[str, Dict[str, Any]] = {}


def tool(
    name: str,
    category: str,
    purpose: str,
    params: Optional[Dict[str, Any]] = None,
):
    """Decorator that registers a tool in the global registry.

    Usage::

        @tool("hash_file", "security", "Compute SHA-256 of a file")
        def hash_file(path: str) -> Dict[str, Any]:
            ...
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        TOOLS[name] = {
            "name": name,
            "category": category,
            "purpose": purpose,
            "params": params or {},
            "fn": fn,
            "signature": str(inspect.signature(fn)),
        }
        return fn

    return deco


# ---------------------------------------------------------------------------
# Registration of the tool families. Each module registers its tools via the
# ``tool`` decorator at import time. We import them lazily but once.
# ---------------------------------------------------------------------------
_LOADED: set = set()


def _ensure_loaded() -> None:
    """Import every tool module exactly once (idempotent).

    Missing optional families are skipped so the toolkit still works in
    environments where only a subset of modules was deployed.
    """
    for mod in (
        "aweai.tools.core",
        "aweai.tools.security",
        "aweai.tools.devops",
        "aweai.tools.devops_tools",
        "aweai.tools.datascience",
        "aweai.tools.media",
        "aweai.tools.automation",
        "aweai.tools.networking",
        "aweai.tools.aiagents",
        "aweai.tools.codegen",
        "aweai.tools.testing",
        "aweai.tools.monitoring",
        "aweai.tools.creative",
        "aweai.tools.mega",
    ):
        if mod not in _LOADED:
            try:
                importlib.import_module(mod)
            except Exception:  # noqa: BLE001  (optional family missing)
                pass
            _LOADED.add(mod)


def list_tools(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return metadata for all registered tools (optionally filtered)."""
    _ensure_loaded()
    out = []
    for name, meta in sorted(TOOLS.items()):
        if category and meta["category"] != category:
            continue
        out.append(
            {
                "name": name,
                "category": meta["category"],
                "purpose": meta["purpose"],
                "params": meta["params"],
            }
        )
    return out


def list_categories() -> List[Dict[str, Any]]:
    """Return categories with per-category tool counts."""
    _ensure_loaded()
    counts: Dict[str, int] = {}
    for meta in TOOLS.values():
        counts[meta["category"]] = counts.get(meta["category"], 0) + 1
    return [
        {"category": cat, "count": counts[cat]}
        for cat in sorted(counts)
    ]


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Return tool metadata by name (or None)."""
    _ensure_loaded()
    return TOOLS.get(name)


def run_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
    """Execute a tool by name with kwargs.

    Returns a normalized result dict:
        {"ok": True, "tool": name, "result": <fn output>}
    or raises KeyError / ValueError for unknown tools.
    """
    _ensure_loaded()
    meta = TOOLS.get(name)
    if meta is None:
        raise KeyError(f"Unknown tool: {name}")
    try:
        result = meta["fn"](**kwargs)
    except TypeError as e:
        raise ValueError(
            f"Tool '{name}' got invalid arguments: {e}. Expected signature: {meta['signature']}"
        ) from e
    if isinstance(result, dict):
        return {"ok": True, "tool": name, "result": result}
    return {"ok": True, "tool": name, "result": {"value": result}}


def tool_count() -> int:
    """Total number of registered tools (after loading all families)."""
    _ensure_loaded()
    return len(TOOLS)


def tool_names(category: Optional[str] = None) -> List[str]:
    """All registered tool names, optionally filtered by category."""
    return [t["name"] for t in list_tools(category)]


__all__ = [
    "TOOLS",
    "tool",
    "list_tools",
    "list_categories",
    "get_tool",
    "run_tool",
    "tool_count",
    "tool_names",
]
