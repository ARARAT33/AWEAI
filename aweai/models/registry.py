"""Registry facade: catalog + installed/custom models on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from aweai.models import MODELS, get_model, list_models
from aweai.config import ensure_runtime_dirs


class ModelRegistry:
    """Merges the built-in catalog with user-installed local models."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        dirs = ensure_runtime_dirs()
        self.models_dir = Path(models_dir or dirs["models"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def catalog(self) -> List[Dict]:
        return list(MODELS)

    def installed(self) -> List[Dict]:
        """Scan the local models dir for saved models (with metadata.json)."""
        found = []
        if not self.models_dir.exists():
            return found
        for child in sorted(self.models_dir.iterdir()):
            meta = child / "metadata.json"
            if child.is_dir() and meta.exists():
                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    data.setdefault("path", str(child))
                    data.setdefault("local", True)
                    found.append(data)
                except (OSError, json.JSONDecodeError):
                    continue
        return found

    def all(self) -> List[Dict]:
        return self.catalog() + self.installed()

    def resolve(self, model_id: str) -> Optional[Dict]:
        m = get_model(model_id)
        if m:
            return m
        for inst in self.installed():
            if inst.get("id") == model_id or inst.get("name") == model_id:
                return inst
        return None

    def register_local(self, name: str, path: str, family: str = "custom",
                       params_b: float = 0.0, metadata: Optional[Dict] = None) -> Dict:
        meta = {
            "id": name,
            "name": name,
            "family": family,
            "params_b": params_b,
            "path": path,
            "local": True,
            "created": metadata.get("created") if metadata else None,
        }
        if metadata:
            meta.update(metadata)
        meta_path = Path(path) / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta
