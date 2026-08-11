# MultiTurn

Multi-turn conversation support maintains context across multiple exchanges.

## Usage

```python
from aweai.compat.multiturn import MultiTurnConversation

conversation = MultiTurnConversation(provider="openai")
response1 = conversation.send("Hello")
response2 = conversation.send("Tell me more")
```

## Related Pages

- [ChatCompletions](ChatCompletions.md) — Chat completions
- [SystemPrompts](SystemPrompts.md) — System prompts
