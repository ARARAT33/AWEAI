# IPU

IPU management handles Graphcore Intelligence Processing Unit resource allocation and scheduling.

## Usage

```bash
# List IPUs
aweai cluster ipus

# Allocate IPU
aweai cluster ipu-allocate --node node1 --ipus 2 --job my_job
```

```python
from aweai.cluster.ipu import IPUManager

ipu = IPUManager()
ipu.allocate(node="node1", ipus=2, job="my_job")
```

## Related Pages

- [GPU](GPU.md) — GPU management
- [Hardware/IPU](../Hardware/IPU.md) — IPU hardware
