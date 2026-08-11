# Databricks

Databricks unified analytics platform support.

## Usage

```python
from aweai.data.formats import DatabricksFormat

db = DatabricksFormat(workspace_url="https://adb-xxx.azuredatabricks.net")
df = db.read_table("database.table")
```

## Related Pages

- [BigQuery](BigQuery.md) — BigQuery
- [Spark](Spark.md) — Spark
- [Formats](Formats.md) — Data formats
