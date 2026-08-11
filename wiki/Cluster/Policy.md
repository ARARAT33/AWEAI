# Policy

Policies define rules and constraints for cluster resource management and job scheduling.

## Usage

```bash
# Create policy
aweai cluster policy create gpu-policy --require-gpu

# Apply policy
aweai cluster policy apply gpu-policy --namespace default
```

```python
from aweai.cluster.policy import PolicyManager

pm = PolicyManager()
pm.create("gpu-policy", rules={"require_gpu": True})
pm.apply("gpu-policy", namespace="default")
```

## Related Pages

- [Admission](Admission.md) — Admission
- [Validator](Validator.md) — Validators
