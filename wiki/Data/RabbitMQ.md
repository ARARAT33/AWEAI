# RabbitMQ

RabbitMQ message broker support.

## Usage

```python
from aweai.data.rabbitmq import RabbitMQClient

rmq = RabbitMQClient(host="localhost", port=5672)
channel = rmq.connect()
channel.basic_publish(exchange="", routing_key="queue_name", body="message")
```

## Related Pages

- [Pulsar](Pulsar.md) — Pulsar
- [ActiveMQ](ActiveMQ.md) — ActiveMQ
- [Kafka](Kafka.md) — Kafka
