# Priority

Priority classes determine the scheduling order and resource allocation for jobs.

## Usage

```bash
# Set job priority
aweai cluster priority set my_job --class high

# List priority classes
aweai cluster priority list
```

```python
from aweai.cluster.priority import PriorityManager

pm = PriorityManager()
pm.set("my_job", priority_class="high")
classes = pm.list()
```

## Related Pages

- [Preemption](Preemption.md) — Preemption
| [Scheduling](Scheduling.md) | Scheduling |
