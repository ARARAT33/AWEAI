# TPU

TPU management handles Tensor Processing Unit resource allocation and optimization.

## Usage

```bash
# Check TPU info
aweai hardware tpu

# Configure TPU
aweai hardware tpu configure --type v4
```

```python
from aweai.hardware.tpu import TPUManager

tpu = TPUManager()
tpu.configure(type="v4")
```

## Related Pages

- [GPU](GPU.md) — GPU
- [NPU](NPU.md) — NPU
- [Tier](Tier.md) — Resource tier
