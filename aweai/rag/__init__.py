"""RAG: retrieval-augmented generation without Hugging Face.

Embeds documents with a hash bag-of-words vectorizer, indexes them to disk,
retrieves the most relevant chunks, and optionally grounds a simple generator.
"""

from .engine import RAGEngine, RAGConfig

__all__ = ["RAGEngine", "RAGConfig"]
