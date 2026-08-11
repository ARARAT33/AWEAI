# ABTesting

A/B testing compares model outputs or providers.

## Usage

```python
from aweai.compat.ab_testing import ABTest

ab = ABTest(providers=["openai", "anthropic"])
results = ab.run(messages=[...], traffic_split=0.5)
```

## Related Pages

- [Router](Router.md) — Provider router
- [CostTracking](CostTracking.md) — Cost tracking
