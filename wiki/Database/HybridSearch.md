# HybridSearch

Hybrid search combines vector similarity with keyword filtering for comprehensive search.

## Usage

```bash
# Hybrid search
aweai db search --type hybrid --query "machine learning tutorial" \
  --vector-query [0.1,0.2,...] --filters "category=tutorial"
```

```python
from aweai.database.hybrid import HybridSearch

search = HybridSearch(vector_index="embeddings", metadata_db="metadata")
results = search.search(
    query="machine learning tutorial",
    vector=[0.1, 0.2, ...],
    filters={"category": "tutorial"}
)
```

## Related Pages

- [Vector](Vector.md) — Vector database
- [Similarity](Similarity.md) — Similarity search
