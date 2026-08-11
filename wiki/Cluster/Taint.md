# Taint

Taints mark nodes to repel pods that don't have matching tolerations.

## Usage

```bash
# Add taint
aweai cluster taint add node1 --key gpu --value true --effect NoSchedule

# Remove taint
aweai cluster taint remove node1 --key gpu
```

```python
from aweai.cluster.taint import TaintManager

tm = TaintManager()
tm.add("node1", key="gpu", value="true", effect="NoSchedule")
tm.remove("node1", key="gpu")
```

## Related Pages

- [Affinity](Affinity.md) — Affinity
| [Toleration](Toleration.md) | Tolerations |
