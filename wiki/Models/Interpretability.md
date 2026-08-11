# Interpretability

Model interpretability focuses on understanding the internal workings and decision-making processes of AI models.

## Approaches

| Approach | Description |
|----------|-------------|
| White-box | Interpretable by design (linear, tree) |
| Post-hoc | Explain black-box models |
| Intrinsic | Built-in interpretability |
| Example-based | Similar examples explanation |

## Usage

```python
from aweai.models.interpretability import Interpreter

interpreter = Interpreter(model)

# Decision boundary
interpreter.plot_decision_boundary(X_test, y_test)

# Feature interactions
interpreter.analyze_interactions(X_test)

# Concept activation
interpreter.concept_activations(X_test, concepts=["cat", "dog"])
```

## Related Pages

- [Explainability](Explainability.md) — Model explainability
- [Fairness](Fairness.md) — Model fairness
- [Safety](Safety.md) — Model safety
