# InfiniBand

InfiniBand is a high-throughput, low-latency networking fabric for HPC and AI clusters.

## Usage

```bash
# Check InfiniBand status
aweai cluster infiniband status

# Configure InfiniBand
aweai cluster infiniband configure --switch ib-switch-1
```

```python
from aweai.cluster.infiniband import InfiniBandManager

ib = InfiniBandManager()
status = ib.status()
ib.configure(switch="ib-switch-1")
```

## Related Pages

- [Network](Network.md) — Network configuration
- [RoCE](RoCE.md) — RoCE
- [Topology](Topology.md) — Topology
