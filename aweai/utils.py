"""Small shared utilities: file helpers, hashing, text chunking, JSON I/O."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
    "for", "with", "is", "are", "was", "were", "be", "been", "by", "as",
    "it", "its", "this", "that", "from", "up", "down", "out", "over",
    "ը", "է", "եւ", "և", "որ", "մի", "այն", "այս", "համար", "դեպի", "վրա",
    "и", "в", "на", "с", "по", "для", "не", "что", "это",
    "le", "la", "les", "un", "une", "de", "du", "des", "et",
    "der", "die", "das", "und", "mit", "für",
    "el", "la", "los", "las", "y", "de", "del",
    "il", "lo", "la", "e", "di", "del",
    "o", "a", "os", "as", "de", "do", "da",
    "的", "了", "在", "是", "和", "就", "不",
    "の", "に", "は", "を", "が",
    "의", "에", "는", "이", "가",
    "ve", "bir", "bu", "şu", "için",
}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    """Rough multilingual tokenizer: words + CJK characters + Armenian."""
    tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]|[\u0560-\u058f]", text.lower())
    return [t for t in tokens if t and t not in STOP_WORDS]


def cosine_similarity(a: List[str], b: List[str]) -> float:
    """Cosine similarity between two token bags (no numpy dependency)."""
    if not a or not b:
        return 0.0
    counts: dict = {}
    for t in set(a):
        counts[t] = counts.get(t, 0) + 1
    score = 0.0
    for t in set(b):
        if t in counts:
            score += counts[t]
    return score / ((len(set(a)) ** 0.5) * (len(set(b)) ** 0.5)) if a and b else 0.0


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # try to break at a sentence or word boundary
            for sep in (". ", "! ", "? ", "\n", " "):
                idx = text.rfind(sep, start + size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        parts.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
        if start >= len(text):
            break
    return [p for p in parts if p]


def flatten(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for it in items:
        if isinstance(it, (list, tuple)):
            out.extend(flatten(it))
        else:
            out.append(it)
    return out


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return name or "model"


def format_bytes(num: Optional[float]) -> str:
    if num is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def truncate(text: str, limit: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
