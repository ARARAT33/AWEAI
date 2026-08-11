# Index

Index management creates and maintains database indexes for query optimization.

## Usage

```bash
# Create index
aweai db index create --table models --column type --type btree

# List indexes
aweai db index list --table models
```

```python
from aweai.database.index import IndexManager

im = IndexManager()
im.create("models", column="type", type="btree")
indexes = im.list("models")
```

## Related Pages

- [Query](Query.md) — Query
- [Sharding](Sharding.md) — Sharding
