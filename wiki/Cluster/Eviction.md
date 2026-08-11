# Eviction

Eviction handles the removal of pods from nodes when resources are constrained.

## Usage

```bash
# Evict pod
aweai cluster evict pod my_pod --grace-period 30

# View evicted pods
aweai cluster eviction list
```

```python
from aweai.cluster.eviction import EvictionManager

em = EvictionManager()
em.evict("my_pod", grace_period=30)
evicted = em.list_evicted()
```

## Related Pages

- [Preemption](Preemption.md) — Preemption
| [Drain](Drain.md) | Drain |
