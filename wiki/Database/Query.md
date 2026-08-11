# Query

Query interface provides a unified way to execute database queries across different backends.

## Usage

```bash
# Execute query
aweai db query --sql "SELECT * FROM models WHERE type='transformer'"

# Query builder
aweai db query --table models --where "type=transformer" --limit 10
```

```python
from aweai.database.query import QueryBuilder

qb = QueryBuilder("models")
results = qb.where("type", "transformer").limit(10).execute()
```

## Related Pages

- [Schema](Schema.md) — Schema
- [Index](Index.md) — Index
