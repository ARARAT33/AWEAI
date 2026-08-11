# Downsampling

Downsampling reduces data resolution for efficient storage and querying.

## Usage

```bash
# Downsample time-series data
aweai db ts downsample --name metrics --from 1s --to 1m --aggregation avg
```

```python
from aweai.database.downsampling import Downsampler

ds = Downsampler()
ds.downsample("metrics", from_resolution="1s", to_resolution="1m", aggregation="avg")
```

## Related Pages

- [TimeSeries](TimeSeries.md) — Time-series database
- [Retention](Retention.md) — Retention
