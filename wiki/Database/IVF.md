# IVF

Inverted File Index (IVF) partitions vectors into clusters for fast approximate nearest neighbor search.

## Usage

```bash
# Create IVF index
aweai db vector create --name embeddings --index-type ivf --dim 768 --nlist 100
```

```python
from aweai.database.vector import IVFIndex

index = IVFIndex(dim=768, nlist=100)
index.train(vectors)
index.add(vectors)
results = index.search(query, k=10)
```

## Related Pages

- [HNSW](HNSW.md) — HNSW index
- [Flat](Flat.md) — Flat index
