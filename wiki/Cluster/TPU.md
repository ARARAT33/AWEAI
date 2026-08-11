# TPU

TPU management handles Tensor Processing Unit resource allocation and job scheduling.

## Usage

```bash
# List TPUs
aweai cluster tpus

# Allocate TPU
aweai cluster tpu-allocate --type v4 --job my_job
```

```python
from aweai.cluster.tpu import TPUManager

tpu = TPUManager()
tpu.allocate(type="v4", job="my_job")
```

## Related Pages

- [GPU](GPU.md) — GPU management
- [Hardware/TPU](../Hardware/TPU.md) — TPU hardware
