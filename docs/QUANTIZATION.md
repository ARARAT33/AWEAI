# Quantization (v2.2)

Quantize any zoo model to run faster and smaller on edge devices.

```bash
aweai quantize my_model --fmt int8        # float16 | int8 | uint8 | int4
```

Result JSON includes:

- `format` — the quantization format
- `original_bytes` / `quantized_bytes` / `compression_ratio`
- `evaluation` — mean/max weight absolute error introduced by quantization

## Python API

```python
from aweai.quantize import quantize_model, load_quantized, list_quantized

res = quantize_model("my_model", fmt="int8")
model, artifact = load_quantized("my_model", fmt="int8")
print(list_quantized())
```

## Edge export

```bash
aweai export-edge my_model --fmt tflite            # onnx|tflite|torchscript|edge_json
aweai export-edge my_model --fmt onnx --quantize int8
aweai edge-footprint my_model                      # fp32/fp16/int8 bytes estimate
```

- ONNX / TorchScript require `torch` (optional dependency).
- TFLite export is dependency-free: it writes a documented JSON artifact
  (`*.tflite.json`) with a reference loader
  (`aweai.export.edge.load_tflite_json`) so it is always usable offline.
- `edge-footprint` reports parameter count and fp32/fp16/int8 byte sizes
  plus an `edge_ready` flag (fits in 64 MB).
