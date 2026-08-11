# OffloadNVMe

NVMe offloading moves model states to high-speed NVMe storage for training very large models with minimal performance impact.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --offload-optimizer-nvme --nvme-path /mnt/nvme
```

```python
from aweai.scale.offloading import NVMeOffloadConfig

config = NVMeOffloadConfig(
    offload_optimizer=True,
    offload_params=True,
    nvme_path="/mnt/nvme",
    async_offload=True
)
```

## Benefits

- High bandwidth for model states
- Minimal training slowdown
- Enables trillion-parameter models

## Related Pages

- [Offloading](Offloading.md) — Offloading overview
- [OffloadCPU](OffloadCPU.md) — CPU offloading
- [OffloadSSD](OffloadSSD.md) — SSD offloading
