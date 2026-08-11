# OpenVINO

OpenVINO optimization accelerates model inference on Intel hardware.

## Usage

```python
from aweai.compat.openvino import OpenVINOModel

model = OpenVINOModel(ir_path="model.xml")
output = model.run(input_data)
```

## Related Pages

- [ONNX](ONNX.md) — ONNX
- [Intel](../AI/Intel.md) — Intel
