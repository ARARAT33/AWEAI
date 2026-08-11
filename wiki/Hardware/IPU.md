# IPU

IPU management handles Graphcore Intelligence Processing Unit resource allocation.

## Usage

```bash
# Check IPU info
aweai hardware ipu

# Configure IPU
aweai hardware ipu configure --ipus 2
```

```python
from aweai.hardware.ipu import IPUManager

ipu = IPUManager()
ipu.configure(ipus=2)
```

## Related Pages

- [GPU](GPU.md) — GPU
- [NPU](NPU.md) — NPU
- [Tier](Tier.md) — Resource tier
