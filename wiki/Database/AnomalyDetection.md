# AnomalyDetection

Anomaly detection identifies unusual patterns in data stored in the database.

## Usage

```bash
# Run anomaly detection
aweai db anomaly detect --name metrics --method isolation_forest
```

```python
from aweai.database.anomaly import AnomalyDetector

ad = AnomalyDetector()
anomalies = ad.detect("metrics", method="isolation_forest")
```

## Related Pages

- [TimeSeries](TimeSeries.md) — Time-series database
- [Monitoring](../Models/Monitoring.md) — Model monitoring
