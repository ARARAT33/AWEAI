# RoCE

RDMA over Converged Ethernet (RoCE) enables high-performance networking over standard Ethernet infrastructure.

## Usage

```bash
# Check RoCE status
aweai cluster roce status

# Configure RoCE
aweai cluster roce configure --priority 3
```

```python
from aweai.cluster.roce import RoCEManager

roce = RoCEManager()
status = roce.status()
roce.configure(priority=3)
```

## Related Pages

- [Network](Network.md) — Network configuration
- [InfiniBand](InfiniBand.md) — InfiniBand
- [Topology](Topology.md) — Topology
