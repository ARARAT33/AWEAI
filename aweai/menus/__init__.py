"""Megamenus package (v3.0)."""

from .catalog import (
    AUTOMATIONS,
    BASE_COMMANDS,
    CATEGORIES,
    build_automations,
    build_catalog,
    catalog_stats,
    render_catalog,
    search_catalog,
)

__all__ = [
    "AUTOMATIONS", "BASE_COMMANDS", "CATEGORIES",
    "build_automations", "build_catalog", "catalog_stats",
    "render_catalog", "search_catalog",
]
