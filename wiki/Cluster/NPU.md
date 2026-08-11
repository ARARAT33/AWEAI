# NPU

NPU management handles Neural Processing Unit resource allocation and scheduling.

## Usage

```bash
# List NPUs
aweai cluster npus

# Allocate NPU
aweai cluster npu-allocate --node node1 --npus 4 --job my_job
```

```python
from aweai.cluster.npu import NPUManager

npu = NPUManager()
npu.allocate(node="node1", npus=4, job="my_job")
```

## Related Pages

- [GPU](GPU.md) — GPU management
- [Hardware/NPU](../Hardware/NPU.md) — NPU hardware
