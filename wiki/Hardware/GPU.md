# GPU

GPU management handles graphics processing unit resource allocation, monitoring, and optimization.

## Usage

```bash
# Check GPU info
aweai hardware gpu

# Monitor GPU
aweai hardware gpu status
```

```python
from aweai.hardware.gpu import GPUManager

gpu = GPUManager()
info = gpu.get_info()
gpu.set_memory_limit(memory="12GB")
```

## Related Pages

- [CPU](CPU.md) — CPU
- [TPU](TPU.md) — TPU
- [Tier](Tier.md) — Resource tier
