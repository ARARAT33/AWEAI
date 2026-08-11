# Compound

Compound architectures combine multiple architectural patterns into unified designs.

## Usage

```python
from aweai.architecture.compound import CompoundModel

compound = CompoundModel(
    blocks=[
        {"type": "transformer", "layers": 6},
        {"type": "moe", "num_experts": 8},
        {"type": "transformer", "layers": 6}
    ]
)

compound.fit(X_train, y_train)
```

## Related Pages

- [StackedModel](StackedModel.md) — Stacked models
- [CascadeModel](CascadeModel.md) — Cascade models
- [EnsembleRouter](EnsembleRouter.md) — Ensemble router
