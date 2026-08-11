# Retention

Retention policies control how long data is kept in databases.

## Usage

```bash
# Set retention
aweai db retention set --name metrics --period 30d

# Apply retention
aweai db retention apply --name metrics
```

```python
from aweai.database.retention import RetentionManager

rm = RetentionManager()
rm.set("metrics", period="30d")
rm.apply("metrics")
```

## Related Pages

- [Downsampling](Downsampling.md) — Downsampling
- [Backup](Backup.md) — Backup
