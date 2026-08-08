# Export

`aweai.export` writes models in four formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| `json` | `.json` | full meta + state, portable |
| `raw` | `.npz` | numpy weight arrays |
| `onnx` | `.onnx` | requires torch + onnx |
| `torchscript` | `.pt` | requires torch |

```bash
aweai export --name m1 --fmt json
aweai export --name m1 --fmt raw
aweai export --name m1 --fmt onnx
```

```python
from aweai.management import export_model
out = export_model("m1", fmt="onnx")
```
