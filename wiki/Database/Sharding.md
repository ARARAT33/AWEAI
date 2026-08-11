# Sharding

Sharding distributes data across multiple database instances for horizontal scaling.

## Usage

```bash
# Enable sharding
aweai db sharding enable --strategy hash --shards 4
```

```python
from aweai.database.sharding import ShardingManager

sm = ShardingManager()
sm.enable(strategy="hash", shards=4)
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `hash` | Hash-based sharding |
| `range` | Range-based sharding |
| `geo` | Geographic sharding |
| `directory` | Directory-based sharding |

## Related Pages

- [Partitioning](Partitioning.md) — Partitioning
- [Replication](Replication.md) — Replication
