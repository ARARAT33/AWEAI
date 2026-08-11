# NVLink

NVLink is a high-bandwidth, low-latency interconnect for GPU-to-GPU communication.

## Usage

```bash
# Check NVLink status
aweai cluster nvlink status

# Configure NVLink topology
aweai cluster nvlink configure --topology fully-connected
```

```python
from aweai.cluster.nvlink import NVLinkManager

nvlink = NVLinkManager()
status = nvlink.status()
nvlink.configure(topology="fully-connected")
```

## Related Pages

- [Network](Network.md) — Network configuration
- [Topology](Topology.md) — Topology
- [GPU](GPU.md) — GPU management
