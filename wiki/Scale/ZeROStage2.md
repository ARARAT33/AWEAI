# ZeROStage2

ZeRO Stage 2 shards both optimizer states and gradients, reducing memory usage by approximately 8x.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --zero-stage 2 --workers 4
```

```python
from aweai.scale.zero import ZeroStage2Config

config = ZeroStage2Config(
    offload_optimizer=False,
    overlap_comm=True
)
```

## Benefits

- 8x memory reduction
- Efficient for large models
- Supports gradient accumulation

## Related Pages

- [ZeRO](ZeRO.md) — Zero Redundancy Optimizer
- [ZeROStage1](ZeROStage1.md) — ZeRO Stage 1
- [ZeROStage3](ZeROStage3.md) — ZeRO Stage 3
