# Export

Data export saves data to various formats and destinations.

## Usage

```bash
# Export to CSV
aweai data export --format csv --output data.csv

# Export to S3
aweai data export --format parquet --output s3://bucket/data.parquet
```

```python
from aweai.data.export import DataExporter

exporter = DataExporter()
exporter.export_to(X, format="csv", path="data.csv")
```

## Related Pages

- [Import](Import.md) — Data import
- [Formats](Formats.md) — Data formats
