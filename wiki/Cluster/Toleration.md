# Toleration

Tolerations allow pods to be scheduled on tainted nodes.

## Usage

```bash
# Add toleration
aweai cluster toleration add my_job --key gpu --operator Equal --value true
```

```python
from aweai.cluster.toleration import TolerationManager

tm = TolerationManager()
tm.add("my_job", key="gpu", operator="Equal", value="true")
```

## Related Pages

- [Taint](Taint.md) — Taints
| [Affinity](Affinity.md) | Affinity |
