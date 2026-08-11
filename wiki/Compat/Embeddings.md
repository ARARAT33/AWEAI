# Embeddings

Embeddings API for generating vector representations.

## Usage

```python
from aweai.compat.embeddings import Embeddings

embeddings = Embeddings(provider="openai")
vectors = embeddings.create(texts=["Hello", "World"])
```

## Related Pages

- [Vector](../Database/Vector.md) — Vector database
- [ChatCompletions](ChatCompletions.md) — Chat completions
