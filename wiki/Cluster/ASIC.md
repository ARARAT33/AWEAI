# ASIC

ASIC management handles Application-Specific Integrated Circuit resource allocation and scheduling.

## Usage

```bash
# List ASICs
aweai cluster asics

# Allocate ASIC
aweai cluster asic-allocate --type cerebras --job my_job
```

```python
from aweai.cluster.asic import ASICManager

asic = ASICManager()
asic.allocate(type="cerebras", job="my_job")
```

## Related Pages

- [GPU](GPU.md) — GPU management
- [Hardware/ASIC](../Hardware/ASIC.md) — ASIC hardware
