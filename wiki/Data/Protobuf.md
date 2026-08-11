# Protobuf

Protocol Buffers serialization support.

## Usage

```python
from aweai.data.protobuf import ProtobufSerializer

pb = ProtobufSerializer(schema_file="schema.proto")
data = pb.serialize(message)
parsed = pb.deserialize(data)
```

## Related Pages

- [Thrift](Thrift.md) — Thrift
- [FlatBuffers](FlatBuffers.md) — FlatBuffers
- [CapnProto](CapnProto.md) — Cap'n Proto
