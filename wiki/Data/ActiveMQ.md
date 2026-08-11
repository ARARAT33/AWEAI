# ActiveMQ

Apache ActiveMQ message broker support.

## Usage

```python
from aweai.data.activemq import ActiveMQClient

amq = ActiveMQClient(broker_url="tcp://localhost:61616")
producer = amq.create_producer("queue_name")
producer.send("message")
```

## Related Pages

- [RabbitMQ](RabbitMQ.md) — RabbitMQ
- [NATS](NATS.md) — NATS
- [Kafka](Kafka.md) — Kafka
