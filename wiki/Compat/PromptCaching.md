# PromptCaching

Prompt caching reduces costs by caching common prompt prefixes.

## Usage

```python
from aweai.compat.prompt_caching import PromptCache

cache = PromptCache(provider="openai")
response = cache.chat(messages=[...], cache_key="system_prompt")
```

## Related Pages

- [KVCache](KVCache.md) — KV cache
- [CostTracking](CostTracking.md) — Cost tracking
