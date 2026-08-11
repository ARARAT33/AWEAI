# FallbackChains

Fallback chains provide automatic failover between providers.

## Usage

```python
from aweai.compat.fallback import FallbackChain

chain = FallbackChain([
    "openai",
    "anthropic",
    "google"
])

response = chain.route(messages=[...])
```

## Related Pages

- [Router](Router.md) — Provider router
- [Providers](Providers.md) — Provider management
