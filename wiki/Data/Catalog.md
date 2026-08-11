# Catalog

Data catalog provides a searchable inventory of datasets.

## Usage

```python
from aweai.data.catalog import DataCatalog

catalog = DataCatalog()
catalog.register("dataset_1", path="./data/dataset_1.csv")
results = catalog.search("type=image")
```

## Related Pages

- [Versioning](Versioning.md) — Data versioning
- [Metadata](Metadata.md) — Data metadata
