# OffloadSSD

SSD offloading moves model states to SSD storage when GPU memory is insufficient, enabling training of extremely large models.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --offload-optimizer-ssd --ssd-path /mnt/ssd
```

```python
from aweai.scale.offloading import SSDOffloadConfig

config = SSDOffloadConfig(
    offload_optimizer=True,
    offload_params=True,
    ssd_path="/mnt/ssd",
    cache_size="2GB"
)
```

## Benefits

- Train models larger than CPU+GPU memory
- Cost-effective for large models
- Supports very large batch sizes

## Related Pages

- [Offloading](Offloading.md) — Offloading overview
- [OffloadCPU](OffloadCPU.md) — CPU offloading
- [OffloadNVMe](OffloadNVMe.md) — NVMe offloading
