# HNSW

Hierarchical Navigable Small World (HNSW) is an efficient approximate nearest neighbor index for vector search.

## Usage

```bash
# Create HNSW index
aweai db vector create --name embeddings --index-type hnsw --dim 768 --M 32
```

```python
from aweai.database.vector import HNSWIndex

index = HNSWIndex(dim=768, M=32)
index.add(vectors)
results = index.search(query, k=10)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `M` | Connections per node |
| `efConstruction` | Build-time accuracy |
| `efSearch` | Search-time accuracy |

## Related Pages

- [IVF](IVF.md) — IVF index
- [Flat](Flat.md) — Flat index
