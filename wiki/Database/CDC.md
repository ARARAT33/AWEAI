# CDC

Change Data Capture (CDC) tracks and propagates data changes in real-time.

## Usage

```bash
# Enable CDC
aweai db cdc enable --table models --output kafka

# View changes
aweai db cdc tail --table models
```

```python
from aweai.database.cdc import CDCManager

cdc = CDCManager()
cdc.enable("models", output="kafka")
changes = cdc.tail("models")
```

## Related Pages

- [Replication](Replication.md) — Replication
- [ETL](ETL.md) — ETL
