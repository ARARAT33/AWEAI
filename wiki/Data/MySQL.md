# MySQL

MySQL database support.

## Usage

```python
from aweai.data.formats import MySQLFormat

mysql = MySQLFormat(host="localhost", port=3306)
data = mysql.query("SELECT * FROM table")
```

## Related Pages

- [PostgreSQL](PostgreSQL.md) — PostgreSQL
- [MongoDB](MongoDB.md) — MongoDB
- [Formats](Formats.md) — Data formats
