"""Data loaders: CSV / JSON / JSONL / text / images -> Dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from aweai.errors import DataError


@dataclass
class Dataset:
    """In-memory dataset with optional X / y / texts / images."""

    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    texts: Optional[List[str]] = None
    images: Optional[np.ndarray] = None
    names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        if self.X is not None:
            return len(self.X)
        if self.texts is not None:
            return len(self.texts)
        if self.images is not None:
            return len(self.images)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": len(self),
            "has_X": self.X is not None,
            "has_y": self.y is not None,
            "has_texts": self.texts is not None,
            "has_images": self.images is not None,
            "X_shape": list(self.X.shape) if self.X is not None else None,
            "y_shape": list(self.y.shape) if self.y is not None else None,
            "names": self.names,
            "meta": self.meta,
        }


def _coerce_target(y: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(y)
    if arr.dtype.kind in ("U", "O", "S"):
        uniq = list(dict.fromkeys(y))  # preserve order
        mapping = {v: i for i, v in enumerate(uniq)}
        return np.array([mapping[v] for v in y], dtype=np.int64), list(uniq)
    return arr.astype(float), None


def load_csv(
    path: Union[str, Path],
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DataError(f"CSV not found: {path}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows: List[Dict[str, Any]] = list(reader)
            if not rows:
                raise DataError(f"CSV is empty: {path}")
            cols = list(rows[0].keys())
        else:
            reader = csv.reader(f, delimiter=delimiter)
            raw = list(reader)
            if not raw:
                raise DataError(f"CSV is empty: {path}")
            cols = [f"col_{i}" for i in range(len(raw[0]))]
            rows = [dict(zip(cols, r)) for r in raw]

    if feature_columns is None:
        feature_columns = [c for c in cols if c != target_column]
    if target_column is not None and target_column not in cols:
        raise DataError(f"target column '{target_column}' not in CSV columns {cols}")

    def num(v: Any) -> float:
        try:
            return float(str(v).strip())
        except Exception:
            return 0.0

    X = np.array([[num(r[c]) for c in feature_columns] for r in rows], dtype=float)
    y = None
    names: List[str] = []
    if target_column is not None:
        y_arr, label_names = _coerce_target([r[target_column] for r in rows])
        y = y_arr
        names = label_names or []
    ds = Dataset(X=X, y=y, names=names)
    ds.meta = {"source": str(p), "columns": feature_columns, "target": target_column, "n": len(rows)}
    return ds


def load_json(path: Union[str, Path], text_key: str = "text", label_key: Optional[str] = "label") -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DataError(f"JSON not found: {path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise DataError(f"Invalid JSON: {e}") from e
    if isinstance(data, dict):
        data = data.get("records", data.get("data", []))
    if not isinstance(data, list):
        raise DataError(f"JSON must be a list of records, got {type(data)}")
    texts = [str(r.get(text_key, "")) for r in data if isinstance(r, dict)]
    y = None
    names: List[str] = []
    if label_key and data and isinstance(data[0], dict) and label_key in data[0]:
        y_arr, label_names = _coerce_target([r.get(label_key) for r in data])
        y = y_arr
        names = label_names or []
    ds = Dataset(texts=texts, y=y, names=names)
    ds.meta = {"source": str(p), "n": len(data), "text_key": text_key, "label_key": label_key}
    return ds


def load_jsonl(path: Union[str, Path], text_key: str = "text", label_key: Optional[str] = "label") -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DataError(f"JSONL not found: {path}")
    texts: List[str] = []
    labels: List[Any] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise DataError(f"Invalid JSONL line: {e}") from e
            if isinstance(obj, dict):
                texts.append(str(obj.get(text_key, "")))
                if label_key is not None and label_key in obj:
                    labels.append(obj[label_key])
            else:
                texts.append(str(obj))
    y = None
    names: List[str] = []
    if labels:
        y_arr, label_names = _coerce_target(labels)
        y = y_arr
        names = label_names or []
    ds = Dataset(texts=texts, y=y, names=names)
    ds.meta = {"source": str(p), "n": len(texts), "text_key": text_key, "label_key": label_key}
    return ds


def load_text(path: Union[str, Path]) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DataError(f"Text file not found: {path}")
    text = p.read_text(encoding="utf-8")
    ds = Dataset(texts=[text])
    ds.meta = {"source": str(p), "n": 1, "chars": len(text)}
    return ds


def load_images(path: Union[str, Path]) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DataError(f"Image dir not found: {path}")
    try:
        from PIL import Image

        _HAS_PIL = True
    except Exception:
        _HAS_PIL = False

    files = sorted([f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")])
    if not files:
        raise DataError(f"No image files found in {path}")
    imgs: List[np.ndarray] = []
    for f in files:
        try:
            if _HAS_PIL:
                im = Image.open(f).convert("L")
                arr = np.asarray(im, dtype=np.float32) / 255.0
            else:
                raise DataError("PIL is required to load images")
            imgs.append(arr.reshape(-1))
        except Exception as e:
            raise DataError(f"Failed to load image {f}: {e}") from e
    arr = np.stack(imgs) if imgs else np.zeros((0, 1))
    ds = Dataset(images=arr)
    ds.meta = {"source": str(p), "n": len(files), "size": arr.shape}
    return ds


def load_any(path: Union[str, Path], **kwargs) -> Dataset:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return load_csv(p, **kwargs)
    if suffix == ".json":
        return load_json(p, **kwargs)
    if suffix in (".jsonl", ".ndjson"):
        return load_jsonl(p, **kwargs)
    if suffix == ".txt":
        return load_text(p)
    if p.is_dir():
        return load_images(p)
    raise DataError(f"Unsupported data file: {path}")
