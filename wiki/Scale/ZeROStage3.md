# ZeROStage3

ZeRO Stage 3 shards optimizer states, gradients, and model parameters, reducing memory usage proportional to the number of data-parallel processes.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --zero-stage 3 --workers 8
```

```python
from aweai.scale.zero import ZeroStage3Config

config = ZeroStage3Config(
    offload_optimizer=True,
    offload_params=True,
    overlap_comm=True
)
```

## Benefits

- n-way memory reduction
- Enables training very large models
- Supports CPU/NVMe offloading

## Related Pages

- [ZeRO](ZeRO.md) — Zero Redundancy Optimizer
- [ZeROStage1](ZeROStage1.md) — ZeRO Stage 1
- [ZeROStage2](ZeROStage2.md) — ZeRO Stage 2
