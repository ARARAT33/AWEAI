# HBase

Apache HBase distributed database support.

## Usage

```python
from aweai.data.formats import HBaseFormat

hbase = HBaseFormat(zookeeper_quorum="localhost")
result = hbase.get("table", "row_key")
```

## Related Pages

- [Cassandra](Cassandra.md) — Cassandra
- [DynamoDB](DynamoDB.md) — DynamoDB
- [Formats](Formats.md) — Data formats
