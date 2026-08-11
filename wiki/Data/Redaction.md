# Redaction

Redaction removes sensitive information from text.

## Usage

```python
from aweai.data.redaction import Redactor

redactor = Redactor(patterns=["email", "phone"])
X_redacted = redactor.redact(texts)
```

## Related Pages

- [Masking](Masking.md) — Data masking
- [Privacy](Privacy.md) — Data privacy
