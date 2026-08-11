# Flink

Apache Flink stream processing support.

## Usage

```python
from aweai.data.flink import FlinkProcessor

flink = FlinkProcessor(env="local")
flink.read_csv("data.csv").print()
```

## Related Pages

- [Spark](Spark.md) — Spark
- [Kafka](Kafka.md) — Kafka
- [Hadoop](Hadoop.md) — Hadoop
