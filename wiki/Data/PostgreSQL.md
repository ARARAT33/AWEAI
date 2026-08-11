# PostgreSQL

PostgreSQL database support.

## Usage

```python
from aweai.data.formats import PostgreSQLFormat

pg = PostgreSQLFormat(host="localhost", port=5432)
data = pg.query("SELECT * FROM table")
```

## Related Pages

- [SQLite](SQLite.md) — SQLite format
- [MySQL](MySQL.md) — MySQL
- [Formats](Formats.md) — Data formats
