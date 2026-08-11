# TensorRT

TensorRT optimization accelerates model inference on NVIDIA GPUs.

## Usage

```python
from aweai.compat.tensorrt import TensorRTModel

model = TensorRTModel(onnx_path="model.onnx")
output = model.run(input_data)
```

## Related Pages

- [ONNX](ONNX.md) — ONNX
- [GPU](../Hardware/GPU.md) — GPU hardware
