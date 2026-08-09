"""Megamenus package (v3.0)."""

from .catalog import (
    AUTOMATIONS,
    BASE_COMMANDS,
    CATEGORIES,
    build_automations,
    build_catalog,
    build_catalog_v31,
    catalog_stats,
    render_catalog,
    search_catalog,
)

__all__ = [
    "AUTOMATIONS", "BASE_COMMANDS", "CATEGORIES",
    "build_automations", "build_catalog", "build_catalog_v31", "catalog_stats",
    "render_catalog", "search_catalog",
]
