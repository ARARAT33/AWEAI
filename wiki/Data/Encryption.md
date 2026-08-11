# Encryption

Data encryption protects data at rest and in transit.

## Usage

```python
from aweai.data.encryption import DataEncryption

enc = DataEncryption(algorithm="AES-256")
X_enc = enc.encrypt(X)
X_dec = enc.decrypt(X_enc)
```

## Related Pages

- [Privacy](Privacy.md) — Data privacy
- [Masking](Masking.md) — Data masking
