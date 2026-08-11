# GPU

GPU management handles GPU resource allocation, monitoring, and scheduling across cluster nodes.

## Usage

```bash
# List GPUs
aweai cluster gpus

# Allocate GPU
aweai cluster gpu-allocate --node node1 --gpus 2 --job my_job

# Monitor GPU
aweai cluster gpu-status node1
```

```python
from aweai.cluster.gpu import GPUManager

gpu = GPUManager()
gpus = gpu.list_available()
gpu.allocate(node="node1", gpus=2, job="my_job")
```

## Related Pages

- [Node](Node.md) — Node management
- [Topology](Topology.md) — Topology
- [NVLink](NVLink.md) — NVLink
