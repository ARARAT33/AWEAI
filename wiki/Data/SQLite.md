# SQLite

SQLite database format support.

## Usage

```python
from aweai.data.formats import SQLiteFormat

sqlite = SQLiteFormat(path="data.db")
data = sqlite.query("SELECT * FROM table")
sqlite.execute("INSERT INTO table VALUES (...)")
```

## Related Pages

- [HDF5](HDF5.md) — HDF5 format
- [PostgreSQL](PostgreSQL.md) — PostgreSQL
- [Formats](Formats.md) — Data formats
