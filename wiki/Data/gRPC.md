# gRPC

gRPC remote procedure call framework support.

## Usage

```python
from aweai.data.grpc import GRPCClient

grpc = GRPCClient(channel="localhost:50051")
response = grpc.call_method("Service.Method", request={...})
```

## Related Pages

- [ZMQ](ZMQ.md) — ZeroMQ
- [Thrift](Thrift.md) — Thrift
- [Protobuf](Protobuf.md) — Protocol Buffers
