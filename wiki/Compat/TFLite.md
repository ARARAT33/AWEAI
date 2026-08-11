# TFLite

TensorFlow Lite enables model deployment on mobile and embedded devices.

## Usage

```python
from aweai.compat.tflite import TFLiteModel

model = TFLiteModel(path="model.tflite")
output = model.run(input_data)
```

## Related Pages

- [CoreML](CoreML.md) — CoreML
- [Edge](../Models/Edge.md) — Edge deployment
