# Smile

Smile binary JSON format support.

## Usage

```python
from aweai.data.smile import SmileSerializer

smile = SmileSerializer()
data = smile.serialize(obj)
parsed = smile.deserialize(data)
```

## Related Pages

- [UBJSON](UBJSON.md) — UBJSON
- [ION](ION.md) — ION
- [JSON](JSON.md) — JSON format
