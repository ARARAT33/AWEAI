# Kafka

Apache Kafka streaming platform support.

## Usage

```python
from aweai.data.kafka import KafkaClient

kafka = KafkaClient(bootstrap_servers="localhost:9092")
consumer = kafka.create_consumer("topic_name")
for msg in consumer:
    process(msg)
```

## Related Pages

- [Flink](Flink.md) — Flink
- [Pulsar](Pulsar.md) — Pulsar
- [RabbitMQ](RabbitMQ.md) — RabbitMQ
