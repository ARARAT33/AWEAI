# Replication

Replication maintains multiple copies of data for high availability and fault tolerance.

## Usage

```bash
# Enable replication
aweai db replication enable --mode master-slave --replicas 3
```

```python
from aweai.database.replication import ReplicationManager

rm = ReplicationManager()
rm.enable(mode="master-slave", replicas=3)
```

## Related Pages

- [Consistency](Consistency.md) — Consistency
- [Backup](Backup.md) — Backup
