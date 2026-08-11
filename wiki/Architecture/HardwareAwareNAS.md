# HardwareAwareNAS

Hardware-Aware Neural Architecture Search optimizes architectures for specific hardware constraints like latency and memory.

## Usage

```python
from aweai.architecture.nas import HardwareAwareNAS

nas = HardwareAwareNAS(
    search_space="transformer",
    hardware="gpu",
    latency_budget="10ms",
    memory_budget="8GB"
)

best_arch = nas.search(X_train, y_train, X_val, y_val)
```

## Related Pages

- [NAS](NAS.md) — Neural architecture search
- [AutoDesigner](AutoDesigner.md) — Auto designer
- [Hardware](../Hardware/Overview.md) — Hardware overview
