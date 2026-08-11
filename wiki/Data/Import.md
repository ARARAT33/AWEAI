# Import

Data import loads data from external sources.

## Usage

```bash
# Import from S3
aweai collect import --source s3://bucket/data.csv

# Import from URL
aweai collect import --source https://example.com/data.json
```

```python
from aweai.data.import_ import DataImporter

importer = DataImporter()
data = importer.import_from("s3://bucket/data.csv")
```

## Related Pages

- [Export](Export.md) — Data export
- [Loaders](Loaders.md) — Data loaders
