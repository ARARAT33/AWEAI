# Feather

Feather format support.

## Usage

```python
from aweai.data.formats import FeatherFormat

feather = FeatherFormat()
data = feather.read("data.feather")
feather.write(data, "output.feather")
```

## Related Pages

- [Parquet](Parquet.md) — Parquet format
- [Formats](Formats.md) — Data formats
