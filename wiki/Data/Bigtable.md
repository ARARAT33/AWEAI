# Bigtable

Google Bigtable distributed database support.

## Usage

```python
from aweai.data.formats import BigtableFormat

bigtable = BigtableFormat(project_id="my-project", instance_id="my-instance")
row = bigtable.read_row("table", "row_key")
```

## Related Pages

- [DynamoDB](DynamoDB.md) — DynamoDB
- [Snowflake](Snowflake.md) — Snowflake
- [Formats](Formats.md) — Data formats
