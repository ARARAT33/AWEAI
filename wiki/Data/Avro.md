# Avro

Apache Avro format support.

## Usage

```python
from aweai.data.formats import AvroFormat

avro = AvroFormat()
data = avro.read("data.avro")
avro.write(data, "output.avro")
```

## Related Pages

- [Parquet](Parquet.md) — Parquet format
- [ORC](ORC.md) — ORC format
- [Formats](Formats.md) — Data formats
