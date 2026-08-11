# OffloadCPU

CPU offloading moves optimizer states or parameters from GPU to CPU during training to reduce GPU memory pressure.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --offload-optimizer-cpu
```

```python
from aweai.scale.offloading import CPUOffloadConfig

config = CPUOffloadConfig(
    offload_optimizer=True,
    pin_memory=True,
    fast_cpu_copy=True
)
```

## Benefits

- Train larger models on limited GPU memory
- Minimal impact on training speed (with NVLink)
- Automatic paging between CPU/GPU

## Related Pages

- [Offloading](Offloading.md) — Offloading overview
- [OffloadSSD](OffloadSSD.md) — SSD offloading
- [OffloadNVMe](OffloadNVMe.md) — NVMe offloading
