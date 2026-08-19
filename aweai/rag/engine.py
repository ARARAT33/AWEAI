"""RAG engine with zero Hugging Face dependency.

Embeddings: hash bag-of-words (numpy). Index: JSON on disk.

FIX (index_file shadowing bug): the old code stored the on-disk index path in
an attribute called `index_file`, and the loaded index dict in a *different*
attribute also called `index_file` on reload, shadowing the path. Here we use
distinct names: `index_path` (the file path) and `_index` (the dict).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aweai.config import ensure_runtime_dirs
from aweai.errors import RAGError
from aweai.utils import chunk_text, cosine_similarity, tokenize, write_json


class RAGConfig:
    def __init__(self, dim: int = 128, chunk_size: int = 500, overlap: int = 50, top_k: int = 3):
        self.dim = int(dim)
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.top_k = int(top_k)

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.dim, "chunk_size": self.chunk_size, "overlap": self.overlap, "top_k": self.top_k}


class RAGEngine:
    """Hash BoW RAG. `index_path` is the on-disk file; `_index` is the dict."""

    def __init__(self, index_path: Optional[str] = None, config: Optional[RAGConfig] = None,
                 data_dir: Optional[str] = None) -> None:
        self.config = config or RAGConfig()
        if data_dir is not None:
            index_path = str(Path(data_dir) / "index.json")
        self.index_path = Path(index_path) if index_path else ensure_runtime_dirs()["rag"] / "index.json"
        self._index: Dict[str, Any] = {"documents": [], "chunks": [], "vectors": []}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._index = data
            except Exception:
                self._index = {"documents": [], "chunks": [], "vectors": []}

    def clear(self) -> None:
        """Reset the index (and remove the on-disk file if present)."""
        self._index = {"documents": [], "chunks": [], "vectors": []}
        try:
            if self.index_path.exists():
                self.index_path.unlink()
        except OSError:
            pass
        return None

    def index_file(self, path: str) -> int:
        """Index a single text file; returns the number of chunks added."""
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        return int(self.index_documents([text])["chunks"])

    # ------------------------------------------------------------- embedding
    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.config.dim, dtype=float)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.config.dim
            sign = 1.0 if (h // self.config.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    # ---------------------------------------------------------------- index
    def index_documents(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None, overwrite: bool = False) -> Dict[str, Any]:
        if overwrite:
            self._index = {"documents": [], "chunks": [], "vectors": []}
        added = 0
        for i, text in enumerate(texts):
            doc_id = str(ids[i]) if ids is not None else f"doc_{len(self._index['documents'])}"
            chunks = chunk_text(str(text), size=self.config.chunk_size, overlap=self.config.overlap)
            for c in chunks:
                self._index["chunks"].append({"doc_id": doc_id, "text": c})
                self._index["vectors"].append(self._embed(c).tolist())
            self._index["documents"].append({"id": doc_id, "chars": len(str(text))})
            added += 1
        self.save()
        return {"indexed": added, "chunks": len(self._index["chunks"]), "path": str(self.index_path)}

    def index_directory(self, path: str, pattern: str = "*.txt") -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise RAGError(f"Directory not found: {path}")
        texts = []
        ids = []
        for f in sorted(p.glob(pattern)):
            try:
                texts.append(f.read_text(encoding="utf-8"))
                ids.append(f.name)
            except Exception:
                continue
        if not texts:
            raise RAGError(f"No {pattern} files found in {path}")
        return self.index_documents(texts, ids=ids, overwrite=True)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> Dict[str, Any]:
        self._load_if_exists()
        return self._index

    # ------------------------------------------------------------- hybrid search & reranking
    def _bm25_score(self, query_tokens: List[str], chunk_text: str, avg_doc_len: float = 100.0, k1: float = 1.2, b: float = 0.75) -> float:
        """Lightweight BM25 lexical relevance score."""
        tokens = tokenize(chunk_text)
        if not tokens or not query_tokens:
            return 0.0
        doc_len = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for q_tok in query_tokens:
            if q_tok in tf:
                freq = tf[q_tok]
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                score += numerator / max(0.001, denominator)
        return min(1.0, score / max(1.0, len(query_tokens)))

    def deduplicate_chunks(self, threshold: float = 0.95) -> Dict[str, Any]:
        """Removes duplicate or near-duplicate document chunks based on embedding similarity."""
        vectors = [np.array(v) for v in self._index["vectors"]]
        if not vectors:
            return {"removed": 0, "remaining": 0}

        keep_indices = []
        for i, vec in enumerate(vectors):
            is_dup = False
            for keep_i in keep_indices:
                sim = float(cosine_similarity(vec, vectors[keep_i]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)

        removed_count = len(vectors) - len(keep_indices)
        if removed_count > 0:
            self._index["chunks"] = [self._index["chunks"][i] for i in keep_indices]
            self._index["vectors"] = [self._index["vectors"][i] for i in keep_indices]
            self.save()

        return {"removed": removed_count, "remaining": len(keep_indices)}

    def search(self, query: str, top_k: Optional[int] = None, hybrid: bool = True, alpha: float = 0.6) -> List[Dict[str, Any]]:
        """Hybrid retrieval: Dense BoW cosine similarity + BM25 lexical reranking."""
        q_vec = self._embed(query)
        q_tokens = tokenize(query)
        k = top_k or self.config.top_k
        scored = []

        avg_len = float(np.mean([len(tokenize(c["text"])) for c in self._index["chunks"]])) if self._index["chunks"] else 100.0

        for i, vec in enumerate(self._index["vectors"]):
            dense_sim = max(0.0, float(cosine_similarity(q_vec, vec)))
            if hybrid:
                bm25_sim = self._bm25_score(q_tokens, self._index["chunks"][i]["text"], avg_doc_len=avg_len)
                combined_score = alpha * dense_sim + (1.0 - alpha) * bm25_sim
            else:
                combined_score = dense_sim
            scored.append((combined_score, dense_sim, i))

        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, dense_sim, i in scored[:k]:
            chunk = self._index["chunks"][i]
            out.append({
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "score": round(float(score), 4),
                "dense_score": round(float(dense_sim), 4),
            })
        return out

    def ask(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        hits = self.search(query, top_k=top_k)
        context = "\n\n".join(h["text"] for h in hits)
        return {"query": query, "answer": context, "sources": hits, "count": len(hits)}

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": len(self._index["documents"]),
            "chunks": len(self._index["chunks"]),
            "vectors": len(self._index["vectors"]),
            "path": str(self.index_path),
        }
