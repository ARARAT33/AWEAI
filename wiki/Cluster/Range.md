# Range

Range defines the minimum and maximum resource bounds for pods.

## Usage

```bash
# Set range
aweai cluster range set my_pod --min-cpu 1 --max-cpu 8
```

```python
from aweai.cluster.range import RangeManager

rm = RangeManager()
rm.set("my_pod", min_cpu=1, max_cpu=8)
```

## Related Pages

- [Limit](Limit.md) — Limits
| [Policy](Policy.md) | Policy |
