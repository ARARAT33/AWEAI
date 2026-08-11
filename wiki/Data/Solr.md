# Solr

Apache Solr search platform support.

## Usage

```python
from aweai.data.formats import SolrFormat

solr = SolrFormat(url="http://localhost:8983/solr")
results = solr.search("my_collection", q="query")
```

## Related Pages

- [Elasticsearch](Elasticsearch.md) — Elasticsearch
- [Cassandra](Cassandra.md) — Cassandra
- [Formats](Formats.md) — Data formats
