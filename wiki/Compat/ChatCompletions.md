# ChatCompletions

Chat completions API for conversational AI.

## Usage

```bash
aweai integrations chat --provider openai --message "Hello"
```

```python
from aweai.compat.chat import ChatCompletions

completions = ChatCompletions(provider="openai")
response = completions.create(messages=[...])
```

## Related Pages

- [Completions](Completions.md) — Text completions
- [OpenAI](OpenAI.md) — OpenAI compatibility
