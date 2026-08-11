# ZMQ

ZeroMQ messaging library support.

## Usage

```python
from aweai.data.zmq import ZMQClient

zmq = ZMQClient(socket_type=zmq.PUB)
zmq.bind("tcp://*:5555")
zmq.send(b"message")
```

## Related Pages

- [NATS](NATS.md) — NATS
- [gRPC](gRPC.md) — gRPC
- [Kafka](Kafka.md) — Kafka
