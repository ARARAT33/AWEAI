# MySQL

MySQL is a popular relational database for production deployments.

## Usage

```bash
# Connect to MySQL
aweai db connect --type mysql --host localhost --port 3306

# Query
aweai db query --sql "SELECT * FROM models"
```

```python
from aweai.database.mysql import MySQLDB

db = MySQLDB(host="localhost", port=3306)
results = db.query("SELECT * FROM models")
```

## Related Pages

- [PostgreSQL](PostgreSQL.md) — PostgreSQL
- [SQLite](SQLite.md) — SQLite
