# Consistency

Consistency models define the guarantees for data reads and writes in distributed databases.

## Usage

```bash
# Set consistency level
aweai db consistency set --level strong
```

```python
from aweai.database.consistency import ConsistencyManager

cm = ConsistencyManager()
cm.set(level="strong")
```

## Levels

| Level | Description |
|-------|-------------|
| `strong` | Immediate consistency |
| `eventual` | Eventual consistency |
| `session` | Session consistency |
| `read-your-writes` | Read-your-own-writes |

## Related Pages

- [Replication](Replication.md) — Replication
- [Sharding](Sharding.md) — Sharding
