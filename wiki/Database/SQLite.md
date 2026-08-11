# SQLite

SQLite is a lightweight, file-based database used for local storage and metadata.

## Usage

```bash
# Initialize SQLite database
aweai db init --type sqlite --path ./aweai.db

# Query
aweai db query --sql "SELECT * FROM models"
```

```python
from aweai.database.sqlite import SQLiteDB

db = SQLiteDB(path="./aweai.db")
results = db.query("SELECT * FROM models")
```

## Related Pages

- [PostgreSQL](PostgreSQL.md) — PostgreSQL
- [MySQL](MySQL.md) — MySQL
