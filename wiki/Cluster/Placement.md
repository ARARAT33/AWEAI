# Placement

Placement strategies determine where jobs run on cluster nodes to optimize performance.

## Usage

```bash
# Set placement policy
aweai cluster placement set-policy my_job --strategy spread
```

```python
from aweai.cluster.placement import PlacementManager

pm = PlacementManager()
pm.set_policy("my_job", strategy="spread")
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `spread` | Spread across nodes |
| `pack` | Pack on same node |
| `binpack` | Bin packing |
| `random` | Random placement |

## Related Pages

- [Scheduling](Scheduling.md) — Scheduling
| [Affinity](Affinity.md) | Affinity |
