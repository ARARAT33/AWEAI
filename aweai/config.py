"""Configuration management for AWEAI.

Stores user settings in a JSON file under ~/.aweai/config.json so nothing
is lost between runs. API keys are kept in a separate file (api_keys.json)
with 0600 permissions.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

APP_DIR_NAME = ".aweai"

DEFAULTS: Dict[str, Any] = {
    "language": "en",
    "port": 8888,
    "auto_open_browser": True,
    "default_model": None,  # resolved automatically from hardware when None
    "model_backend": "auto",  # auto | local | api
    "api_provider": "openai",
    "api_base_url": "https://api.openai.com/v1",
    "api_model": "gpt-4o-mini",
    "local_model_dir": None,  # where downloaded/custom models are stored
    "data_dir": None,
    "rag_backend": "json",  # json | chroma | faiss
    "rag_embedding": "hash",  # hash | tfidf | huggingface
    "log_level": "INFO",
    "telemetry": False,
    "max_history": 200,
}


def app_dir() -> Path:
    """Return ~/.aweai creating it if needed."""
    root = Path.home() / APP_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


class Config:
    """Thin JSON-backed settings store."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else app_dir() / "config.json"
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        self._data = dict(DEFAULTS)
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    self._data.update(stored)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def update(self, values: Dict[str, Any]) -> None:
        self._data.update(values)
        self.save()

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def reset(self) -> None:
        self._data = dict(DEFAULTS)
        self.save()


class ApiKeyStore:
    """Stores API keys with 0600 file permissions."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else app_dir() / "api_keys.json"
        self._data: Dict[str, str] = {}

    def _ensure_mode(self) -> None:
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def load(self) -> None:
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    self._data = {k: str(v) for k, v in stored.items()}
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._ensure_mode()

    def get(self, provider: str) -> Optional[str]:
        self.load()
        return self._data.get(provider)

    def set(self, provider: str, key: str) -> None:
        self.load()
        self._data[provider] = key
        self.save()

    def delete(self, provider: str) -> bool:
        self.load()
        if provider in self._data:
            del self._data[provider]
            self.save()
            return True
        return False


def get_config() -> Config:
    return Config()


def get_api_keys() -> ApiKeyStore:
    return ApiKeyStore()


def ensure_runtime_dirs(cfg: Optional[Config] = None) -> Dict[str, Path]:
    cfg = cfg or get_config()
    base = Path(cfg.get("data_dir") or (app_dir() / "data"))
    dirs = {
        "models": Path(cfg.get("local_model_dir") or (base / "models")),
        "rag": base / "rag",
        "actions": base / "actions",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def which_ok(cmd: str) -> bool:
    return shutil.which(cmd) is not None
