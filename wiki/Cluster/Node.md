# Node

Node management handles individual cluster node configuration, health monitoring, and lifecycle.

## Usage

```bash
# List nodes
aweai cluster nodes

# Check node health
aweai cluster node-status node1

# Drain node
aweai cluster drain node1
```

```python
from aweai.cluster.node import NodeManager

nm = NodeManager()
nodes = nm.list_nodes()
health = nm.check_health("node1")
```

## Related Pages

- [Manager](Manager.md) — Cluster manager
- [GPU](GPU.md) — GPU management
