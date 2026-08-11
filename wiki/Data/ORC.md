# ORC

Apache ORC format support.

## Usage

```python
from aweai.data.formats import ORCFormat

orc = ORCFormat()
data = orc.read("data.orc")
orc.write(data, "output.orc")
```

## Related Pages

- [Avro](Avro.md) — Avro format
- [Parquet](Parquet.md) — Parquet format
- [Formats](Formats.md) — Data formats
