"""AWEAI tools — the extension toolkit with hundreds of unique-purpose tools.

This package grows AWEAI into a universal workbench with core, security,
devops, data, media, automation, agent, codegen, testing, monitoring,
creative, mega and Universal AI Ecosystem tools.
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

from aweai.tools import core  # noqa: F401
from aweai.tools import security  # noqa: F401
for _optional_family in (
    "mega", "mega2", "aiagents", "automation", "codegen", "creative", "datascience",
    "devops", "media", "monitoring", "networking", "testing", "ecosystem",
):
    try:
        __import__(f"aweai.tools.{_optional_family}")
    except Exception:
        pass

ALL_TOOLS = TOOLS

__all__ = [
    "TOOLS", "ALL_TOOLS", "tool", "list_tools", "list_categories", "get_tool",
    "run_tool", "tool_count", "tool_names", "core", "security", "devops",
    "datascience", "media", "automation", "networking", "aiagents", "codegen",
    "testing", "monitoring", "creative", "mega", "mega2", "ecosystem",
]
