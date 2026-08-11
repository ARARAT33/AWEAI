# Versioning

Data versioning tracks changes to datasets over time.

## Usage

```python
from aweai.data.versioning import DataVersioning

dv = DataVersioning()
dv.save_version("dataset_v1", X, y)
dv.list_versions()
```

## Related Pages

- [Lineage](Lineage.md) — Data lineage
- [Catalog](Catalog.md) — Data catalog
