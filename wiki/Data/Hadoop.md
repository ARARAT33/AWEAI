# Hadoop

Apache Hadoop distributed storage support.

## Usage

```python
from aweai.data.hadoop import HDFSClient

hdfs = HDFSClient(host="localhost", port=9000)
hdfs.upload("local_file.txt", "/hdfs/path/file.txt")
```

## Related Pages

- [Spark](Spark.md) — Spark
- [Flink](Flink.md) — Flink
- [Kafka](Kafka.md) — Kafka
