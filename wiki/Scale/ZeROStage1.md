# ZeROStage1

ZeRO Stage 1 shards optimizer states across data-parallel processes, reducing memory usage by approximately 4x.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --zero-stage 1 --workers 4
```

```python
from aweai.scale.zero import ZeroStage1Config

config = ZeroStage1Config(
    offload_optimizer=False,
    overlap_comm=True
)
```

## Benefits

- 4x memory reduction for optimizer states
- Minimal communication overhead
- Easy integration with existing code

## Related Pages

- [ZeRO](ZeRO.md) — Zero Redundancy Optimizer
- [ZeROStage2](ZeROStage2.md) — ZeRO Stage 2
- [ZeROStage3](ZeROStage3.md) — ZeRO Stage 3
