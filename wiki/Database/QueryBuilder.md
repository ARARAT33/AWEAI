# QueryBuilder

Query builder provides a fluent interface for constructing database queries.

## Usage

```bash
# Build query
aweai db query-builder from models where type=transformer select name,params limit 10
```

```python
from aweai.database.query_builder import QueryBuilder

qb = QueryBuilder("models")
query = (qb.select("name", "params")
            .where("type", "transformer")
            .limit(10)
            .build())
results = qb.execute()
```

## Related Pages

- [Query](Query.md) — Query
- [ORM](ORM.md) — ORM
