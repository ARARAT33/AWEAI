# MessagePack

MessagePack binary serialization support.

## Usage

```python
from aweai.data.messagepack import MessagePackSerializer

mp = MessagePackSerializer()
data = mp.serialize(obj)
parsed = mp.deserialize(data)
```

## Related Pages

- [CapnProto](CapnProto.md) — Cap'n Proto
- [CBOR](CBOR.md) — CBOR
- [Protobuf](Protobuf.md) — Protocol Buffers
