# Completions

Text completions API for generating text continuations.

## Usage

```python
from aweai.compat.completions import TextCompletions

completions = TextCompletions(provider="openai")
response = completions.create(prompt="Once upon a time")
```

## Related Pages

- [ChatCompletions](ChatCompletions.md) — Chat completions
- [OpenAI](OpenAI.md) — OpenAI compatibility
