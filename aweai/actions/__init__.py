"""Automation: natural-language actions, pipelines, batch jobs."""

from .actions import parse_action, ACTION_PATTERNS
from .runner import run_action, run_pipeline, save_pipeline, list_pipelines, run_batch

__all__ = ["parse_action", "ACTION_PATTERNS", "run_action", "run_pipeline", "save_pipeline", "list_pipelines", "run_batch"]
