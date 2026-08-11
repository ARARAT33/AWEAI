# Limit

Limits constrain resource usage for individual pods or containers.

## Usage

```bash
# Set limits
aweai cluster limit set my_pod --cpu 4 --memory 16GB --gpus 1

# View limits
aweai cluster limit get my_pod
```

```python
from aweai.cluster.limit import LimitManager

lm = LimitManager()
lm.set("my_pod", cpu=4, memory="16GB", gpus=1)
```

## Related Pages

- [Quota](Quota.md) — Quota
| [Range](Range.md) | Range |
