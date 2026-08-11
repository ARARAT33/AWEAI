# Similarity

Similarity search finds the most similar items to a query using distance metrics.

## Usage

```bash
# Search similar
aweai db vector search --name embeddings --query [0.1,0.2,...] --top-k 10 \
  --metric cosine
```

```python
from aweai.database.vector import SimilaritySearch

search = SimilaritySearch(index="embeddings")
results = search.search(query, k=10, metric="cosine")
```

## Metrics

| Metric | Description |
|--------|-------------|
| `cosine` | Cosine similarity |
| `euclidean` | Euclidean distance |
| `dot_product` | Dot product |
| `manhattan` | Manhattan distance |

## Related Pages

- [Vector](Vector.md) — Vector database
- [HNSW](HNSW.md) — HNSW index
