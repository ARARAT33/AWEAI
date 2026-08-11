# BigQuery

Google BigQuery data warehouse support.

## Usage

```python
from aweai.data.formats import BigQueryFormat

bq = BigQueryFormat(project_id="my-project")
df = bq.query("SELECT * FROM `my-project.dataset.table`")
```

## Related Pages

- [Redshift](Redshift.md) — Redshift
- [Databricks](Databricks.md) — Databricks
- [Formats](Formats.md) — Data formats
