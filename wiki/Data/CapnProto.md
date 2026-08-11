# CapnProto

Cap'n Proto serialization support.

## Usage

```python
from aweai.data.capnproto import CapnProtoSerializer

cp = CapnProtoSerializer(schema_file="schema.capnp")
data = cp.serialize(message)
parsed = cp.deserialize(data)
```

## Related Pages

- [FlatBuffers](FlatBuffers.md) — FlatBuffers
- [Protobuf](Protobuf.md) — Protocol Buffers
- [MessagePack](MessagePack.md) — MessagePack
