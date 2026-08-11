# Cordon

Cordon marks a node as unschedulable, preventing new pods from being placed on it.

## Usage

```bash
# Cordon node
aweai cluster cordon node1

# Uncordon node
aweai cluster uncordon node1
```

```python
from aweai.cluster.cordon import CordonManager

cm = CordonManager()
cm.cordon("node1")
cm.uncordon("node1")
```

## Related Pages

- [Drain](Drain.md) — Drain
| [Scheduling](Scheduling.md) | Scheduling |
