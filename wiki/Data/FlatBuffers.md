# FlatBuffers

FlatBuffers serialization support.

## Usage

```python
from aweai.data.flatbuffers import FlatBuffersSerializer

fb = FlatBuffersSerializer(schema_file="schema.fbs")
data = fb.serialize(message)
parsed = fb.deserialize(data)
```

## Related Pages

- [Protobuf](Protobuf.md) — Protocol Buffers
- [CapnProto](CapnProto.md) — Cap'n Proto
- [MessagePack](MessagePack.md) — MessagePack
