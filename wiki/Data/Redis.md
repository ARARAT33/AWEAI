# Redis

Redis in-memory data store support.

## Usage

```python
from aweai.data.formats import RedisFormat

redis = RedisFormat(host="localhost", port=6379)
redis.set("key", "value")
value = redis.get("key")
```

## Related Pages

- [MongoDB](MongoDB.md) — MongoDB
- [Elasticsearch](Elasticsearch.md) — Elasticsearch
- [Formats](Formats.md) — Data formats
