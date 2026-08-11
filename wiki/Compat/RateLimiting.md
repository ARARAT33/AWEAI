# RateLimiting

Rate limiting controls request frequency to providers.

## Usage

```python
from aweai.compat.rate_limiting import RateLimiter

rl = RateLimiter(provider="openai", max_rpm=60)
rl.wait_if_needed()
```

## Related Pages

- [CostTracking](CostTracking.md) — Cost tracking
- [WarmPools](WarmPools.md) — Warm pools
