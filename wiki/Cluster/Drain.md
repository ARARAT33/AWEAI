# Drain

Drain safely removes a node from service by evicting all pods.

## Usage

```bash
# Drain node
aweai cluster drain node1 --grace-period 300 --ignore-daemonsets

# Uncordon node
aweai cluster uncordon node1
```

```python
from aweai.cluster.drain import DrainManager

dm = DrainManager()
dm.drain("node1", grace_period=300)
dm.uncordon("node1")
```

## Related Pages

- [Cordon](Cordon.md) — Cordon
| [Eviction](Eviction.md) | Eviction |
