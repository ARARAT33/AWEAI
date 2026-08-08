"""AWEAI model factory utilities: tokenization, chunking, serialization."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union, Dict

import numpy as np

try:  # pragma: no cover - optional
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_ALNUM_RE = re.compile(r"[\w\u0561-\u0587\u0531-\u0556]+", re.UNICODE)


def truncate(text: str, limit: int = 500) -> str:
    """Truncate text to `limit` characters with an ellipsis marker."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def safe_filename(name: str) -> str:
    """Turn arbitrary text into a safe directory/file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "model"


def tokenize(text: str) -> List[str]:
    """Lightweight word tokenizer (no external deps, supports Armenian/Cyrillic)."""
    return _ALNUM_RE.findall(text.lower())


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by character length (no tokenizers dep)."""
    if size <= 0:
        return [text]
    step = max(size - overlap, 1)
    chunks: List[str] = []
    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        if chunk:
            chunks.append(chunk)
    return chunks


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (numpy fallback included).

    If string tokens are passed (e.g. ["a", "b"]), they are treated as a
    bag-of-words: each token becomes a dimension (order-independent).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind in ("U", "O") or b.dtype.kind in ("U", "O"):
        vocab: Dict[str, int] = {}
        for tok in list(a) + list(b):
            if tok not in vocab:
                vocab[tok] = len(vocab)
        va = np.zeros(len(vocab), dtype=float)
        vb = np.zeros(len(vocab), dtype=float)
        for tok in a:
            va[vocab[tok]] += 1.0
        for tok in b:
            vb[vocab[tok]] += 1.0
        a, b = va, vb
    else:
        a = a.astype(float)
        b = b.astype(float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def write_json(path: Union[str, Path], data: Any) -> None:
    """Write JSON with a serialization hook for numpy/torch objects."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def read_json(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (tuple, set)):
        return list(o)
    if _HAS_TORCH and isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def stable_hash(*parts: Any) -> str:
    """Stable short hash used for model ids / n-gram keys."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- n-gram keys
def serialize_ngram_key(key: Tuple[str, ...]) -> str:
    """Serialize an n-gram tuple key into a JSON-safe string.

    Fixes the classic n-gram tuple-key serialization bug where keys were
    stored as Python tuple repr (e.g. "('the', 'cat')") and broke when the
    tokens contained quotes, or when loading from JSON.
    """
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def deserialize_ngram_key(key: str) -> Tuple[str, ...]:
    """Deserialize an n-gram key serialized by :func:`serialize_ngram_key`."""
    if isinstance(key, tuple):
        return key
    try:
        data = json.loads(key)
        return tuple(str(t) for t in data)
    except Exception:
        # Legacy fallback: Python tuple repr like "('a', 'b')"
        if key.startswith("(") and key.endswith(")"):
            inner = key[1:-1]
            parts = []
            for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"|([^,]+)", inner):
                parts.append(m.group(1) or m.group(2) or m.group(3).strip())
            return tuple(parts)
        return (key,)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
