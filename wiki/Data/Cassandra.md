# Cassandra

Apache Cassandra distributed database support.

## Usage

```python
from aweai.data.formats import CassandraFormat

cassandra = CassandraFormat(contact_points=["localhost"])
results = cassandra.query("SELECT * FROM keyspace.table")
```

## Related Pages

- [Solr](Solr.md) — Solr
- [HBase](HBase.md) — HBase
- [Formats](Formats.md) — Data formats
