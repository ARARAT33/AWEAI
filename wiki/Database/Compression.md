# Compression

Compression reduces storage requirements and improves query performance.

## Usage

```bash
# Enable compression
aweai db compression enable --algorithm zstd --level 3
```

```python
from aweai.database.compression import CompressionManager

cm = CompressionManager()
cm.enable(algorithm="zstd", level=3)
```

## Related Pages

- [Retention](Retention.md) — Retention
- [Replication](Replication.md) — Replication
