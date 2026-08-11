# Redshift

Amazon Redshift data warehouse support.

## Usage

```python
from aweai.data.formats import RedshiftFormat

rs = RedshiftFormat(
    host="cluster.xxx.redshift.amazonaws.com",
    port=5439,
    database="dev",
    user="user",
    password="password"
)
df = rs.query("SELECT * FROM schema.table")
```

## Related Pages

- [Snowflake](Snowflake.md) — Snowflake
- [BigQuery](BigQuery.md) — BigQuery
- [Formats](Formats.md) — Data formats
