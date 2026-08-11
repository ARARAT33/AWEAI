# Sharding

Parameter sharding distributes model parameters across multiple devices to enable training of models larger than single-device memory.

## Usage

```bash
aweai dtrain transformer --name model --data data.csv --target label \
  --sharding strategy=full
```

```python
from aweai.scale.sharding import ShardingConfig

config = ShardingConfig(
    strategy="full",  # full, stage1, stage2, stage3
    shard_size=1024
)
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `full` | Shard all parameters |
| `stage1` | Shard optimizer states |
| `stage2` | Shard optimizer + gradients |
| `stage3` | Shard all + params |

## Related Pages

- [ZeRO](ZeRO.md) — Zero Redundancy Optimizer
- [FSDP](FSDP.md) — Fully Sharded Data Parallel
