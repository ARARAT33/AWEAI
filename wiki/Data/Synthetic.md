# Synthetic

Synthetic data generation creates artificial data for training.

## Usage

```python
from aweai.data.synthetic import SyntheticDataGenerator

generator = SyntheticDataGenerator()
X_synth, y_synth = generator.generate(n_samples=1000)
```

## Related Pages

- [Generation](Generation.md) — Data generation
- [Augment](Augment.md) — Data augmentation
