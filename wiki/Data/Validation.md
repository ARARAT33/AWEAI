# Validation

Data validation ensures data quality and correctness.

## Usage

```python
from aweai.data.validation import DataValidator

validator = DataValidator(schema=schema)
is_valid = validator.validate(X)
errors = validator.get_errors()
```

## Related Pages

- [Deduplication](Deduplication.md) — Deduplication
- [Cleaning](Cleaning.md) — Data cleaning
