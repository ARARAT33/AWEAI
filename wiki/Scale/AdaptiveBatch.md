# AdaptiveBatch

Adaptive batch sizing dynamically adjusts batch sizes during training based on GPU memory availability and performance.

## Usage

```bash
aweai train --type transformer --name model --data data.csv --target label \
  --params '{"adaptive_batch": true, "min_batch": 8, "max_batch": 256}'
```

```python
from aweai.scale.adaptive_batch import AdaptiveBatchConfig

config = AdaptiveBatchConfig(
    enabled=True,
    min_batch=8,
    max_batch=256,
    target_memory_util=0.9
)
```

## Benefits

- Maximizes GPU utilization
- Handles memory pressure automatically
- Reduces manual tuning

## Related Pages

- [GradientAccumulation](GradientAccumulation.md) — Gradient accumulation
- [MixedPrecision](MixedPrecision.md) — Mixed precision
