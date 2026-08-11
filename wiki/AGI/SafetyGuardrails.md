# SafetyGuardrails

Safety guardrails prevent agents from taking harmful actions.

## Usage

```python
from aweai.agi.safety import SafetyGuardrails

guardrails = SafetyGuardrails()
is_safe = guardrails.check(action)
```

## Related Pages

- [SandboxedExecutor](SandboxedExecutor.md) — Sandboxed executor
- [AlignmentChecker](AlignmentChecker.md) — Alignment checking
