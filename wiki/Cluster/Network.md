# Network

Network configuration manages inter-node communication, bandwidth allocation, and network topology in the cluster.

## Usage

```bash
# Configure network
aweai cluster network configure --topology fat-tree --bandwidth 200Gbps

# Check network status
aweai cluster network status
```

```python
from aweai.cluster.network import NetworkManager

nm = NetworkManager()
nm.configure(topology="fat-tree", bandwidth="200Gbps")
status = nm.status()
```

## Related Pages

- [NVLink](NVLink.md) — NVLink
- [InfiniBand](InfiniBand.md) — InfiniBand
- [RoCE](RoCE.md) — RoCE
- [Topology](Topology.md) — Topology
