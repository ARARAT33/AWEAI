# Masking

Data masking obscures sensitive data while preserving utility.

## Usage

```python
from aweai.data.masking import DataMasker

masker = DataMasker()
X_masked = masker.mask(X, columns=["ssn", "credit_card"])
```

## Related Pages

- [Encryption](Encryption.md) — Data encryption
- [Redaction](Redaction.md) — Redaction
