# NATS

NATS messaging system support.

## Usage

```python
from aweai.data.nats import NATSClient

nats = NATSClient(servers=["nats://localhost:4222"])
await nats.connect()
await nats.publish("subject", b"message")
```

## Related Pages

- [ActiveMQ](ActiveMQ.md) — ActiveMQ
- [ZMQ](ZMQ.md) — ZeroMQ
- [Kafka](Kafka.md) — Kafka
