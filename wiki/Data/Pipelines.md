# Pipelines

Data pipelines orchestrate multi-step data processing workflows.

## Usage

```python
from aweai.data.pipelines import Pipeline

pipeline = Pipeline([
    ("load", load_data),
    ("clean", clean_data),
    ("transform", transform_data),
    ("split", split_data)
])

result = pipeline.run()
```

## Related Pages

- [Augment](Augment.md) — Data augmentation
- [Transformation](Transformation.md) — Data transformation
