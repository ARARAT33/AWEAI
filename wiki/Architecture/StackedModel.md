# StackedModel

Stacked models combine multiple base models in a stacked ensemble for improved predictive performance.

## Usage

```python
from aweai.architecture.stacked import StackedModel

stacked = StackedModel(
    base_models=["mlp", "transformer", "lstm"],
    meta_model="linear"
)

stacked.fit(X_train, y_train)
```

## Related Pages

- [EnsembleRouter](EnsembleRouter.md) — Ensemble router
- [CascadeModel](CascadeModel.md) — Cascade models
- [Compound](Compound.md) — Compound architectures
