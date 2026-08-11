# UBJSON

UBJSON (Universal Binary JSON) support.

## Usage

```python
from aweai.data.ubjson import UBJSONSerializer

ubjson = UBJSONSerializer()
data = ubjson.serialize(obj)
parsed = ubjson.deserialize(data)
```

## Related Pages

- [CBOR](CBOR.md) — CBOR
- [Smile](Smile.md) — Smile
- [JSON](JSON.md) — JSON format
