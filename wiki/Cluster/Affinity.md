# Affinity

Affinity rules control how jobs are scheduled relative to other jobs or nodes.

## Usage

```bash
# Set node affinity
aweai cluster affinity set-node my_job --node node1

# Set pod affinity
aweai cluster affinity set-pod my_job --pod my_model --weight 10
```

```python
from aweai.cluster.affinity import AffinityManager

am = AffinityManager()
am.set_node_affinity("my_job", node="node1")
am.set_pod_affinity("my_job", pod="my_model", weight=10)
```

## Related Pages

- [Placement](Placement.md) — Placement
| [Taint](Taint.md) | Taints |
| [Toleration](Toleration.md) | Tolerations |
