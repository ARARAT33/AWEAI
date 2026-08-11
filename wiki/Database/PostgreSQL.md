# PostgreSQL

PostgreSQL is a powerful relational database for production deployments.

## Usage

```bash
# Connect to PostgreSQL
aweai db connect --type postgresql --host localhost --port 5432

# Create table
aweai db query --sql "CREATE TABLE models (id SERIAL PRIMARY KEY, name TEXT)"
```

```python
from aweai.database.postgres import PostgreSQLDB

db = PostgreSQLDB(host="localhost", port=5432)
db.execute("CREATE TABLE models (...)")
```

## Related Pages

- [SQLite](SQLite.md) — SQLite
- [MySQL](MySQL.md) — MySQL
