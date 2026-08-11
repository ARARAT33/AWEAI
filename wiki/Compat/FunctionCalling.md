# FunctionCalling

Function calling enables models to invoke external functions.

## Usage

```python
from aweai.compat.function_calling import FunctionCalling

fc = FunctionCalling(provider="openai")
response = fc.create(
    messages=[...],
    functions=[...],
    function_call="auto"
)
```

## Related Pages

- [ChatCompletions](ChatCompletions.md) — Chat completions
- [Tools](../Tools/Overview.md) — Tools overview
