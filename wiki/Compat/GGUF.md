# GGUF

GGUF format enables efficient model deployment with llama.cpp.

## Usage

```python
from aweai.compat.gguf import GGUFModel

model = GGUFModel(path="model.gguf")
output = model.generate(prompt="Hello")
```

## Related Pages

- [TFLite](TFLite.md) — TensorFlow Lite
- [llama.cpp](../AI/llama.cpp.md) — llama.cpp
