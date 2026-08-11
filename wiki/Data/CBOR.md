# CBOR

CBOR (Concise Binary Object Representation) support.

## Usage

```python
from aweai.data.cbor import CBORSerializer

cbor = CBORSerializer()
data = cbor.serialize(obj)
parsed = cbor.deserialize(data)
```

## Related Pages

- [MessagePack](MessagePack.md) — MessagePack
- [UBJSON](UBJSON.md) — UBJSON
- [Protobuf](Protobuf.md) — Protocol Buffers
