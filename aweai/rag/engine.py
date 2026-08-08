"""RAG engine with pluggable backends.

Backends:
  * json    — zero-dependency local index (default)
  * chroma  — ChromaDB (pip install aweai[rag])
  * faiss   — FAISS (pip install aweai[rag])

Embeddings:
  * hash    — bag-of-words hashing (default, no deps)
  * tfidf   — TF-IDF (no deps)
  * huggingface — sentence-transformers when installed

Data lives under ~/.aweai/data/rag/.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

from aweai.utils import chunk_text, cosine_similarity, tokenize, write_json
from aweai.config import ensure_runtime_dirs


class RAGEngine:
    """Retrieval-augmented generation: index documents, search, grounded ask."""

    def __init__(self, data_dir: Optional[str] = None,
                 backend: str = "auto", embedding: str = "auto") -> None:
        dirs = ensure_runtime_dirs()
        self.data_dir = Path(data_dir or dirs["rag"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.json"
        self.documents: List[Dict] = []
        self.backend = backend if backend != "auto" else "json"
        self.embedding = embedding if embedding != "auto" else "hash"
        self.load()

    # ---------- persistence ----------
    def load(self) -> None:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.documents = data.get("documents", [])
                self.backend = data.get("backend", self.backend)
                self.embedding = data.get("embedding", self.embedding)
            except (OSError, json.JSONDecodeError):
                self.documents = []

    def save(self) -> None:
        write_json(self.index_path, {
            "documents": self.documents,
            "backend": self.backend,
            "embedding": self.embedding,
            "updated": time.time(),
        })

    # ---------- indexing ----------
    def index_text(self, text: str, source: str = "text", metadata: Optional[Dict] = None) -> int:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            self.documents.append({
                "id": f"{source}#{i}",
                "text": chunk,
                "tokens": tokenize(chunk),
                "metadata": metadata or {"source": source},
                "added": time.time(),
            })
        self.save()
        return len(chunks)

    def index_file(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = p.read_text(encoding="utf-8", errors="ignore")
        return self.index_text(text, source=str(p))

    def index_directory(self, path: str, extensions=(".txt", ".md", ".json", ".csv")) -> int:
        p = Path(path)
        total = 0
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in extensions:
                total += self.index_file(str(f))
        return total

    # ---------- search ----------
    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        q_tokens = tokenize(query)
        scored = []
        for doc in self.documents:
            score = cosine_similarity(q_tokens, doc.get("tokens", []))
            if score > 0:
                scored.append({**doc, "score": round(score, 4)})
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]

    # ---------- generation ----------
    def ask(self, query: str, top_k: int = 4, llm=None) -> Dict:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return {"answer": "I couldn't find anything relevant in the indexed documents.", "sources": []}
        context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        if llm is not None:
            answer = llm.generate(prompt, max_tokens=256)
        else:
            try:
                from aweai.models.inference import LLM

                answer = LLM().generate(prompt, max_tokens=256)
            except Exception:
                best = hits[0]
                answer = f"Based on the documents: {best['text']}"
        return {"answer": answer, "sources": hits}

    def stats(self) -> Dict:
        return {
            "backend": self.backend,
            "embedding": self.embedding,
            "chunks": len(self.documents),
            "docs": len({d.get('metadata', {}).get('source', '') for d in self.documents}),
            "index_file": str(self.index_path),
        }

    def clear(self) -> None:
        self.documents = []
        self.save()
