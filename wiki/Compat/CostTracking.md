# CostTracking

Cost tracking monitors API usage and costs across providers.

## Usage

```python
from aweai.compat.cost_tracking import CostTracker

ct = CostTracker()
cost = ct.get_cost(provider="openai", period="30d")
```

## Related Pages

- [ABTesting](ABTesting.md) — A/B testing
- [RateLimiting](RateLimiting.md) — Rate limiting
