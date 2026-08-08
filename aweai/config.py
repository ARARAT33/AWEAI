"""Configuration handling for AWEAI.

Supports loading settings from environment variables, a JSON/TOML-style
config file, or Python defaults.  Values are resolved with a clear
precedence: explicit constructor kwargs > environment > config file >
defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0
DEFAULT_DB_PATH = "aweai.db"


@dataclass
class AWEConfig:
    """Central configuration object for an AWEAI application.

    Attributes:
        api_key: API key for the LLM provider (falls back to env).
        model: Model identifier (e.g. "gpt-4o-mini", "claude-3-5-sonnet").
        base_url: Optional custom OpenAI-compatible endpoint.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per generation.
        timeout: HTTP timeout in seconds.
        db_path: Path to the SQLite memory database.
        system_prompt: Optional system prompt for the agent.
        extra: Free-form extra settings (config file passthrough).
    """

    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    db_path: str = DEFAULT_DB_PATH
    system_prompt: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Environment variable names
    # ------------------------------------------------------------------
    _ENV_API_KEY = "AWEAI_API_KEY"
    _ENV_MODEL = "AWEAI_MODEL"
    _ENV_BASE_URL = "AWEAI_BASE_URL"
    _ENV_DB_PATH = "AWEAI_DB_PATH"

    def __post_init__(self) -> None:
        # Resolve environment fallbacks for fields that were not set.
        if self.api_key is None:
            self.api_key = os.environ.get(self._ENV_API_KEY)
        if self.model == DEFAULT_MODEL:
            self.model = os.environ.get(self._ENV_MODEL, DEFAULT_MODEL)
        if self.base_url is None:
            self.base_url = os.environ.get(self._ENV_BASE_URL)
        if self.db_path == DEFAULT_DB_PATH:
            self.db_path = os.environ.get(self._ENV_DB_PATH, DEFAULT_DB_PATH)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "AWEConfig":
        """Load configuration from a JSON file (optionally with an ``env``
        section that is applied to the process environment before resolution)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

        # Apply env overrides if present.
        for key, value in data.pop("env", {}).items():
            os.environ[key] = str(value)

        # Unknown keys are preserved as ``extra``.
        known = {
            "api_key", "model", "base_url", "temperature",
            "max_tokens", "timeout", "db_path", "system_prompt",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        clean = {k: v for k, v in data.items() if k in known}
        clean["extra"] = extra
        return cls(**clean)

    @classmethod
    def from_env(cls) -> "AWEConfig":
        """Create configuration purely from environment variables."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (useful for debugging/logging)."""
        data = asdict(self)
        # Never log secrets.
        if data.get("api_key"):
            data["api_key"] = "***"
        return data
