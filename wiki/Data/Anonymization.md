# Anonymization

Anonymization removes or obscures personally identifiable information.

## Usage

```python
from aweai.data.anonymization import Anonymizer

anonymizer = Anonymizer()
X_anon = anonymizer.anonymize(X, columns=["name", "email"])
```

## Related Pages

- [Privacy](Privacy.md) — Data privacy
- [Pseudonymization](Pseudonymization.md) — Pseudonymization
