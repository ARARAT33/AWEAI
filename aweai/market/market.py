"""Model marketplace (v3.0).

Publish / download / rate models. The marketplace is a local-first registry:

* ``~/.aweai/market/index.json`` — the marketplace index (listings).
* Each listing stores metadata + a pointer to the packaged model archive
  (zip) or the model zoo entry.

CLI:
    aweai market search <query>
    aweai market publish <name> [--tag v1 --description "..."]
    aweai market download <model_id>
    aweai market rate <model_id> <stars>
    aweai market list
    aweai market info <model_id>

The registry is fully local and dependency-free, so it works offline and on
edge devices. Publishing creates a zip archive of the model zoo entry so it
can be shared and downloaded on another machine.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.config import ensure_runtime_dirs
from aweai.errors import MarketError
from aweai.management.manager import get_model_path
from aweai.utils import read_json, safe_filename, write_json


def _market_path() -> Path:
    return ensure_runtime_dirs()["base"] / "market"


def _index_path() -> Path:
    return _market_path() / "index.json"


def _load_index() -> Dict[str, Any]:
    data = read_json(_index_path())
    if data is None:
        data = {"version": 1, "listings": {}}
    return data


def _save_index(index: Dict[str, Any]) -> None:
    _market_path().mkdir(parents=True, exist_ok=True)
    write_json(_index_path(), index)


def _new_id(name: str, tag: str = "") -> str:
    return f"{safe_filename(name)}-{time.strftime('%Y%m%d')}-{abs(hash(name + tag)) % 10000}"


def publish(name: str, tag: str = "v1", description: str = "", author: str = "") -> Dict[str, Any]:
    """Publish a zoo model to the marketplace (creates a zip archive)."""
    root = get_model_path(name)
    model_json = root / "model.json"
    if not model_json.exists():
        raise MarketError(f"Model '{name}' not found in zoo")
    index = _load_index()
    mid = _new_id(name, tag)
    archive_dir = _market_path() / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{mid}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_json, arcname="model.json")
        version_json = root / "version.json"
        if version_json.exists():
            zf.write(version_json, arcname="version.json")
        qdir = root / "quantized"
        if qdir.exists():
            for f in qdir.glob("*.json"):
                zf.write(f, arcname=f"quantized/{f.name}")
        edir = root / "edge"
        if edir.exists():
            for f in edir.glob("*"):
                if f.is_file():
                    zf.write(f, arcname=f"edge/{f.name}")

    payload = read_json(model_json)
    meta = payload.get("meta", {})
    listing = {
        "id": mid,
        "name": name,
        "tag": tag,
        "description": description or f"Model '{name}' published from AWEAI",
        "author": author or "anonymous",
        "model_type": meta.get("model_type"),
        "version": meta.get("version", 1),
        "metrics": meta.get("metrics", {}),
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "downloads": 0,
        "ratings": [],
        "avg_rating": 0.0,
    }
    index["listings"][mid] = listing
    _save_index(index)
    return {"published": mid, "listing": listing}


def search(query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """Search marketplace listings by name/model_type/description."""
    index = _load_index()
    q = query.lower().strip()
    out = []
    for listing in index.get("listings", {}).values():
        if not q:
            out.append(listing)
            continue
        hay = " ".join([
            listing.get("name", ""),
            listing.get("model_type", ""),
            listing.get("description", ""),
            listing.get("author", ""),
            listing.get("tag", ""),
        ]).lower()
        if q in hay:
            out.append(listing)
    out.sort(key=lambda l: l.get("published_at", ""), reverse=True)
    return out[:limit]


def list_listings() -> List[Dict[str, Any]]:
    return search("", limit=1000)


def info(model_id: str) -> Dict[str, Any]:
    index = _load_index()
    listing = index.get("listings", {}).get(model_id)
    if listing is None:
        raise MarketError(f"Marketplace model '{model_id}' not found")
    return listing


def download(model_id: str, name: Optional[str] = None, as_name: Optional[str] = None) -> Dict[str, Any]:
    """Download a marketplace model into the local zoo."""
    listing = info(model_id)
    archive = Path(listing["archive"])
    if not archive.exists():
        raise MarketError(f"Archive missing for '{model_id}'")
    tmp = _market_path() / "tmp" / model_id
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp)
    model_json = tmp / "model.json"
    if not model_json.exists():
        raise MarketError(f"Archive for '{model_id}' has no model.json")
    payload = read_json(model_json)
    meta = payload.get("meta", {})
    target = safe_filename(as_name or name or listing.get("name") or model_id)

    from aweai.models.registry import create_model, list_model_types
    model_type = meta.get("model_type")
    if model_type not in list_model_types():
        raise MarketError(f"Unsupported model_type in archive: {model_type}")
    model = create_model(model_type, **dict(meta.get("params", {})))
    model.load_state(payload.get("state", {}))
    model.metrics = meta.get("metrics", {})
    model.history = meta.get("history", {"loss": [], "val_loss": []})
    model.trained = True

    from aweai.management import save_model
    saved = save_model(model, target, meta={"model_type": model_type})

    index = _load_index()
    index["listings"][model_id]["downloads"] = int(index["listings"][model_id].get("downloads", 0)) + 1
    _save_index(index)
    return {"downloaded": model_id, "saved_as": target, "saved": saved}


def rate(model_id: str, stars: float, comment: str = "") -> Dict[str, Any]:
    """Rate a marketplace model (1-5 stars)."""
    index = _load_index()
    listing = index.get("listings", {}).get(model_id)
    if listing is None:
        raise MarketError(f"Marketplace model '{model_id}' not found")
    stars = float(stars)
    if stars < 1 or stars > 5:
        raise MarketError("Rating must be between 1 and 5")
    listing.setdefault("ratings", []).append({
        "stars": stars,
        "comment": comment,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    vals = [r["stars"] for r in listing["ratings"]]
    listing["avg_rating"] = round(sum(vals) / len(vals), 2)
    _save_index(index)
    return {"model": model_id, "avg_rating": listing["avg_rating"], "count": len(vals)}


def stats() -> Dict[str, Any]:
    index = _load_index()
    listings = index.get("listings", {})
    return {
        "listings": len(listings),
        "total_downloads": sum(int(l.get("downloads", 0)) for l in listings.values()),
        "avg_rating": round(sum(float(l.get("avg_rating", 0.0)) for l in listings.values()) / max(len(listings), 1), 2),
    }
