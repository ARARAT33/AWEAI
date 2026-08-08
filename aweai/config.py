"""Configuration store: ~/.aweai/config.json + API key store.

Paths:
  ~/.aweai/config.json     — user config (language, port, model…)
  ~/.aweai/api_keys.json   — BYOK provider keys (chmod 0600)
  ~/.aweai/data/           — RAG index
  ~/.aweai/models/         — trained models

Environment overrides:
  AWEAI_HOME   — override the base directory
  AWEAI_LANG   — default language
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "language": "en",
    "port": 8888,
    "auto_open_browser": True,
    "default_model": None,
    "model_backend": "auto",  # auto | local | api
    "api_provider": None,
    "api_base_url": None,
    "api_model": None,
}


class Config:
    """Small JSON-backed config with dot-path get/set."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or (app_dir() / "config.json"))
        self._data: Dict[str, Any] = dict(DEFAULTS)
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass
        # environment overrides
        env_lang = os.environ.get("AWEAI_LANG")
        if env_lang:
            self._data["language"] = env_lang

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def update(self, values: Dict[str, Any]) -> None:
        for k, v in values.items():
            if v is not None:
                self._data[k] = v
        self.save()

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def app_dir() -> Path:
    """Base directory for AWEAI state (~/.aweai by default)."""
    override = os.environ.get("AWEAI_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aweai"


def ensure_runtime_dirs() -> Dict[str, Path]:
    base = app_dir()
    dirs = {
        "base": base,
        "config": base,
        "data": base / "data",
        "rag": base / "data" / "rag",
        "models": base / "models",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


class ApiKeyStore:
    """BYOK provider keys, stored 0600."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or (app_dir() / "api_keys.json"))
        self._data: Dict[str, str] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {k: str(v) for k, v in loaded.items()}
            except (OSError, json.JSONDecodeError):
                pass

    def get(self, provider: str) -> Optional[str]:
        return self._data.get(provider)

    def set(self, provider: str, key: str) -> None:
        self._data[provider] = key
        self.save()

    def all(self) -> Dict[str, str]:
        return dict(self._data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


_config_singleton: Optional[Config] = None
_keys_singleton: Optional[ApiKeyStore] = None


def get_config() -> Config:
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton


def get_api_keys() -> ApiKeyStore:
    global _keys_singleton
    if _keys_singleton is None:
        _keys_singleton = ApiKeyStore()
    return _keys_singleton


def get_platform() -> str:
    import platform

    return platform.system().lower()


def which_ok(cmd: str) -> bool:
    import shutil

    return shutil.which(cmd) is not None
