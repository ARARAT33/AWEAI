# KVCache

KV cache optimization improves inference speed by caching key-value pairs.

## Usage

```python
from aweai.compat.kv_cache import KVCacheManager

cache = KVCacheManager(max_size="4GB")
cache.set("key", tensor)
tensor = cache.get("key")
```

## Related Pages

- [ContinuousBatching](ContinuousBatching.md) — Continuous batching
- [PromptCaching](PromptCaching.md) — Prompt caching
