# CascadeModel

Cascade models pass data through a sequence of models, with each stage refining the output of the previous stage.

## Usage

```python
from aweai.architecture.cascade import CascadeModel

cascade = CascadeModel(
    stages=["transformer", "linear"]
)

cascade.fit(X_train, y_train)
```

## Related Pages

- [StackedModel](StackedModel.md) — Stacked models
- [EnsembleRouter](EnsembleRouter.md) — Ensemble router
- [Compound](Compound.md) — Compound architectures
