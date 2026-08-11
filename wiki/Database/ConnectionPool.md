# ConnectionPool

Connection pooling manages database connections for efficient resource utilization.

## Usage

```bash
# Configure connection pool
aweai db pool configure --min-connections 5 --max-connections 20
```

```python
from aweai.database.connection_pool import ConnectionPool

pool = ConnectionPool(min_connections=5, max_connections=20)
conn = pool.get_connection()
```

## Related Pages

- [ETL](ETL.md) — ETL
- [ORM](ORM.md) — ORM
- [QueryBuilder](QueryBuilder.md) — Query builder
