# Thrift

Apache Thrift RPC framework support.

## Usage

```python
from aweai.data.thrift import ThriftClient

thrift = ThriftClient(host="localhost", port=9090)
response = thrift.call("Service.method", request={...})
```

## Related Pages

- [gRPC](gRPC.md) — gRPC
- [Protobuf](Protobuf.md) — Protocol Buffers
- [Avro](Avro.md) — Avro format
