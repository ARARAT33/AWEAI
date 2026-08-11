# Pulsar

Apache Pulsar messaging system support.

## Usage

```python
from aweai.data.pulsar import PulsarClient

pulsar = PulsarClient(service_url="pulsar://localhost:6650")
consumer = pulsar.subscribe("topic_name", "subscription_name")
```

## Related Pages

- [Kafka](Kafka.md) — Kafka
- [NATS](NATS.md) — NATS
- [RabbitMQ](RabbitMQ.md) — RabbitMQ
