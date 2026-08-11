# ONNX

ONNX compatibility enables cross-platform model deployment.

## Usage

```python
from aweai.compat.onnx import ONNXModel

model = ONNXModel(path="model.onnx")
output = model.run(input_data)
```

## Related Pages

- [Export](../Models/Export.md) — Model export
- [TensorRT](TensorRT.md) — TensorRT
- [OpenVINO](OpenVINO.md) — OpenVINO
