# Pseudonymization

Pseudonymization replaces identifiers with pseudonyms.

## Usage

```python
from aweai.data.pseudonymization import Pseudonymizer

pseudo = Pseudonymizer()
X_pseudo = pseudo.pseudonymize(X, columns=["user_id"])
```

## Related Pages

- [Anonymization](Anonymization.md) — Anonymization
- [Privacy](Privacy.md) — Data privacy
