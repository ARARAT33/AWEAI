# Streaming

Streaming support for real-time response delivery.

## Usage

```python
from aweai.compat.streaming import StreamingCompletions

stream = StreamingCompletions(provider="openai")
for chunk in stream.create(messages=[...]):
    print(chunk, end="")
```

## Related Pages

- [ChatCompletions](ChatCompletions.md) — Chat completions
- [Completions](Completions.md) — Text completions
