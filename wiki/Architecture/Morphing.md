# Morphing

Architecture Morphing gradually transforms one neural network architecture into another during training.

## Usage

```python
from aweai.architecture.morphing import ArchitectureMorpher

morpher = ArchitectureMorpher(
    source="mlp",
    target="transformer",
    schedule="linear"
)

morpher.morph(model, epoch=50)
```

## Related Pages

- [Converter](Converter.md) — Architecture converter
- [WeightTransfer](WeightTransfer.md) — Weight transfer
- [AutoDesigner](AutoDesigner.md) — Auto designer
