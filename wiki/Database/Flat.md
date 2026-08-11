# Flat

Flat index stores vectors without indexing, providing exact nearest neighbor search.

## Usage

```bash
# Create flat index
aweai db vector create --name embeddings --index-type flat --dim 768
```

```python
from aweai.database.vector import FlatIndex

index = FlatIndex(dim=768)
index.add(vectors)
results = index.search(query, k=10)
```

## Related Pages

- [HNSW](HNSW.md) — HNSW index
- [IVF](IVF.md) — IVF index
