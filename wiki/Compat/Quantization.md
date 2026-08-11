# Quantization

Quantization support for provider compatibility.

## Usage

```python
from aweai.compat.quantization import QuantizedModel

qm = QuantizedModel(provider="local", quantization="int4")
response = qm.generate(prompt="Hello")
```

## Related Pages

- [Quantization](../Models/Quantization.md) — Model quantization
- [Edge](../Models/Edge.md) — Edge deployment
