"""AWEAI tools — the extension toolkit with hundreds of unique-purpose tools.

This package grows AWEAI into a universal workbench:

* :mod:`registry`   — tool registry + dispatcher (list/run/describe)
* :mod:`core`       — system, file, process, network and info tools
* :mod:`security`   — hashing, crypto, auditing and scanning tools
* :mod:`devops`     — git, docker, CI, packaging and deploy tools
* :mod:`datascience`— statistics, ML metrics, transforms and text tools
* :mod:`media`      — image, audio, video and OCR helpers
* :mod:`automation` — jobs, workflows, alerts, timers and batch tools
* :mod:`networking` — DNS, HTTP, ping, ports, TLS and web tools
* :mod:`aiagents`   — prompts, chains, memory, evals and RAG helpers
* :mod:`codegen`    — code generation, refactoring, linting and docs
* :mod:`testing`    — pytest, fuzz, coverage, benchmark and assert tools
* :mod:`monitoring` — metrics, traces, logs, health and alert tools
* :mod:`creative`   — ideas, naming, design, content and 100k-menu tools
* :mod:`mega`       — 1000+ declarative stdlib tools (math, str, json, ...)
* :mod:`mega2`      — 697+ more stdlib tools (crypto, ml, web, db, cloud, ...)

Usage::

    from aweai.tools import list_tools, run_tool, tool_count
    print(tool_count())          # e.g. 1800+
    run_tool("hash_sha256", text="hello")
"""

from __future__ import annotations

from aweai.tools.registry import (
    TOOLS,
    get_tool,
    list_categories,
    list_tools,
    run_tool,
    tool,
    tool_count,
    tool_names,
)

# Import all tool families so their @tool decorators register.
# Optional families are guarded so partial deployments still import.
from aweai.tools import core  # noqa: F401  (side-effect registration)
from aweai.tools import security  # noqa: F401
for _optional_family in (
    "mega", "mega2", "aiagents", "automation", "codegen", "creative", "datascience",
    "devops", "media", "monitoring", "networking", "testing",
):
    try:
        __import__(f"aweai.tools.{_optional_family}")
    except Exception:  # noqa: BLE001  (optional family missing)
        pass

# Convenience aggregate: name -> metadata (for UI/CLI rendering).
ALL_TOOLS = TOOLS

__all__ = [
    "TOOLS",
    "ALL_TOOLS",
    "tool",
    "list_tools",
    "list_categories",
    "get_tool",
    "run_tool",
    "tool_count",
    "tool_names",
    "core",
    "security",
    "devops",
    "datascience",
    "media",
    "automation",
    "networking",
    "aiagents",
    "codegen",
    "testing",
    "monitoring",
    "creative",
    "mega",
    "mega2",
]
