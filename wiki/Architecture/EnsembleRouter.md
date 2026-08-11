# EnsembleRouter

Ensemble Router dynamically selects or combines multiple expert models based on input characteristics.

## Usage

```python
from aweai.architecture.ensemble_router import EnsembleRouter

router = EnsembleRouter(
    experts=["mlp", "transformer", "lstm"],
    routing_strategy="soft"
)

router.fit(X_train, y_train)
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `hard` | Select single best expert |
| `soft` | Weighted combination |
| `adaptive` | Learn routing policy |

## Related Pages

- [StackedModel](StackedModel.md) — Stacked models
- [CascadeModel](CascadeModel.md) — Cascade models
- [MoE](MoE.md) — Mixture of Experts
