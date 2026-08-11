# Preemption

Preemption allows higher-priority jobs to evict lower-priority jobs from cluster nodes.

## Usage

```bash
# Enable preemption
aweai cluster preemption enable --policy priority

# View preempted jobs
aweai cluster preemption list
```

```python
from aweai.cluster.preemption import PreemptionManager

pm = PreemptionManager()
pm.enable(policy="priority")
evicted = pm.list_evicted()
```

## Related Pages

- [Priority](Priority.md) — Priority
| [Eviction](Eviction.md) | Eviction |
