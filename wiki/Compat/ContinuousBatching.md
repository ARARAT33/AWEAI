# ContinuousBatching

Continuous batching processes requests as they arrive for optimal throughput.

## Usage

```python
from aweai.compat.continuous_batching import ContinuousBatcher

cb = ContinuousBatcher()
for request in requests:
    cb.add(request)
```

## Related Pages

- [DynamicBatching](DynamicBatching.md) — Dynamic batching
- [KVCache](KVCache.md) — KV cache
