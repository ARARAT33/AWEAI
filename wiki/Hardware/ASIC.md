# ASIC

ASIC management handles Application-Specific Integrated Circuit resource allocation.

## Usage

```bash
# Check ASIC info
aweai hardware asic

# Configure ASIC
aweai hardware asic configure --type cerebras
```

```python
from aweai.hardware.asic import ASICManager

asic = ASICManager()
asic.configure(type="cerebras")
```

## Related Pages

- [GPU](GPU.md) — GPU
- [TPU](TPU.md) — TPU
- [Tier](Tier.md) — Resource tier
