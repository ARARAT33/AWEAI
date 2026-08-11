# ETL

ETL (Extract, Transform, Load) manages data pipelines for database integration.

## Usage

```bash
# Create ETL pipeline
aweai db etl create --name sync_pipeline --source postgresql --target sqlite

# Run pipeline
aweai db etl run sync_pipeline
```

```python
from aweai.database.etl import ETLManager

etl = ETLManager()
etl.create("sync_pipeline", source="postgresql", target="sqlite")
etl.run("sync_pipeline")
```

## Related Pages

- [CDC](CDC.md) — Change data capture
- [ConnectionPool](ConnectionPool.md) — Connection pool
