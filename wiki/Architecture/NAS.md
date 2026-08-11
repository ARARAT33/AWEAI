# NAS

Neural Architecture Search (NAS) automatically searches for optimal neural network architectures.

## Usage

```python
from aweai.architecture.nas import NAS

nas = NAS(
    search_space="transformer",
    metric="accuracy",
    time_budget=3600
)

best_arch = nas.search(X_train, y_train, X_val, y_val)
```

## Related Pages

- [AutoDesigner](AutoDesigner.md) — Auto designer
- [HardwareAwareNAS](HardwareAwareNAS.md) — Hardware-aware NAS
- [AutoML](../Models/AutoML.md) — Automated machine learning
