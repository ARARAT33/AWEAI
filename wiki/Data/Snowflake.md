# Snowflake

Snowflake cloud data warehouse support.

## Usage

```python
from aweai.data.formats import SnowflakeFormat

sf = SnowflakeFormat(
    account="account",
    user="user",
    password="password"
)
df = sf.query("SELECT * FROM database.schema.table")
```

## Related Pages

- [Bigtable](Bigtable.md) — Bigtable
- [Redshift](Redshift.md) — Redshift
- [Formats](Formats.md) — Data formats
