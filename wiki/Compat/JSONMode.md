# JSONMode

JSON mode constrains model output to valid JSON.

## Usage

```python
from aweai.compat.json_mode import JSONMode

jm = JSONMode(provider="openai")
response = jm.create(
    messages=[...],
    response_format={"type": "json_object"}
)
```

## Related Pages

- [ChatCompletions](ChatCompletions.md) — Chat completions
- [Types](Types.md) — Type definitions
