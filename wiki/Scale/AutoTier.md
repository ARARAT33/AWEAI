# AutoTier

Auto-tier automatically manages memory hierarchy by dynamically moving model states between GPU, CPU, and SSD based on access patterns.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --auto-tier
```

```python
from aweai.scale.autotier import AutoTierConfig

config = AutoTierConfig(
    enabled=True,
    gpu_size="24GB",
    cpu_size="64GB",
    ssd_size="500GB"
)
```

## Benefits

- Optimal memory utilization
- Transparent to training code
- Reduces manual tuning

## Related Pages

- [Offloading](Offloading.md) — Offloading overview
- [OffloadCPU](OffloadCPU.md) — CPU offloading
- [OffloadNVMe](OffloadNVMe.md) — NVMe offloading
