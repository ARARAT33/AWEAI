# Partitioning

Partitioning divides large tables into smaller, more manageable pieces.

## Usage

```bash
# Partition table
aweai db partition create --table metrics --column timestamp --by month
```

```python
from aweai.database.partitioning import PartitionManager

pm = PartitionManager()
pm.create("metrics", column="timestamp", by="month")
```

## Related Pages

- [Sharding](Sharding.md) — Sharding
- [Index](Index.md) — Index
