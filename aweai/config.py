"""Configuration store for AWEAI.

Paths:
  ~/.aweai/config.json     — user config (language, port, defaults…)
  ~/.aweai/data/           — datasets & RAG index
  ~/.aweai/models/         — trained models (the model zoo)
  ~/.aweai/logs/           — training logs & curves

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
    "device": "auto",  # auto | cpu | cuda | mps
    "verbosity": "info",
    "dataset_dir": None,
    "model_dir": None,
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
        "pipelines": base / "pipelines",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_platform() -> str:
    import platform

    return platform.system().lower()


def which_ok(cmd: str) -> bool:
    import shutil

    return shutil.which(cmd) is not None


_config_singleton: Optional[Config] = None


def get_config() -> Config:
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton
