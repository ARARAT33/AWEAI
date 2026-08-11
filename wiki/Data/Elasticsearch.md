# Elasticsearch

Elasticsearch search engine support.

## Usage

```python
from aweai.data.formats import ElasticsearchFormat

es = ElasticsearchFormat(hosts=["localhost:9200"])
results = es.search(index="my_index", query={...})
```

## Related Pages

- [Redis](Redis.md) — Redis
- [Solr](Solr.md) — Solr
- [Formats](Formats.md) — Data formats
