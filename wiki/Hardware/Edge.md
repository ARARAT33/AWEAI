# Edge

Edge hardware refers to resource-constrained devices for on-device inference.

## Usage

```bash
# Check edge compatibility
aweai edge-footprint my_model
```

```python
from aweai.hardware.edge import EdgeManager

em = EdgeManager(device="raspberry_pi")
em.deploy(model="my_model", format="tflite")
```

## Related Pages

- [PC](PC.md) — PC
- [ResourceTier](ResourceTier.md) — Resource tier classification
