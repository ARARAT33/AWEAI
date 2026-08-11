# WarmPools

Warm pools maintain pre-initialized connections for low-latency access.

## Usage

```python
from aweai.compat.warm_pools import WarmPool

wp = WarmPool(provider="openai", pool_size=10)
response = wp.get().chat(messages=[...])
```

## Related Pages

- [RateLimiting](RateLimiting.md) — Rate limiting
- [DynamicBatching](DynamicBatching.md) — Dynamic batching
