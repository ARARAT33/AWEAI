# SandboxedExecutor

Sandboxed executor runs untrusted code in an isolated environment for safety.

## Usage

```python
from aweai.agi.sandbox import SandboxedExecutor

executor = SandboxedExecutor()
result = executor.execute(code, timeout=30)
```

## Related Pages

- [SafetyGuardrails](SafetyGuardrails.md) — Safety guardrails
- [Agent](Agent.md) — Agent framework
