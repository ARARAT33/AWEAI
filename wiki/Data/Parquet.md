# Parquet

Apache Parquet columnar storage format support.

## Usage

```python
from aweai.data.formats import ParquetFormat

parquet = ParquetFormat()
data = parquet.read("data.parquet")
parquet.write(data, "output.parquet")
```

## Related Pages

- [TOML](TOML.md) — TOML format
- [Formats](Formats.md) — Data formats
