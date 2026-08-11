# Quota

Quotas limit resource consumption by namespace, project, or user.

## Usage

```bash
# Set quota
aweai cluster quota set namespace1 --cpu 16 --memory 64GB --gpus 4

# View quota
aweai cluster quota get namespace1
```

```python
from aweai.cluster.quota import QuotaManager

qm = QuotaManager()
qm.set("namespace1", cpu=16, memory="64GB", gpus=4)
```

## Related Pages

- [Limit](Limit.md) — Limits
| [Class](Class.md) | Class |
