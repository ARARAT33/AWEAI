# Embedding

Embeddings convert discrete data into continuous vector representations.

## Usage

```python
from aweai.data.embedding import TextEmbedding

embedder = TextEmbedding(model="all-MiniLM-L6-v2")
embeddings = embedder.embed(["Hello world", "Goodbye world"])
```

## Related Pages

- [FeatureEngineering](FeatureEngineering.md) — Feature engineering
- [Vectorization](Vectorization.md) — Vectorization
