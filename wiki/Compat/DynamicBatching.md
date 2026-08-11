# DynamicBatching

Dynamic batching groups requests for efficient processing.

## Usage

```python
from aweai.compat.dynamic_batching import DynamicBatcher

batcher = DynamicBatcher(max_batch_size=32, timeout=0.1)
response = batcher.add(request).execute()
```

## Related Pages

- [ContinuousBatching](ContinuousBatching.md) — Continuous batching
- [WarmPools](WarmPools.md) — Warm pools
