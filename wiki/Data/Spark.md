# Spark

Apache Spark distributed processing support.

## Usage

```python
from aweai.data.spark import SparkProcessor

spark = SparkProcessor(app_name="AWEAI")
df = spark.read.csv("data.csv")
df.show()
```

## Related Pages

- [Databricks](Databricks.md) — Databricks
- [Hadoop](Hadoop.md) — Hadoop
- [Flink](Flink.md) — Flink
