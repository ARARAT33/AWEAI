# Profiling

Data profiling analyzes dataset statistics and characteristics.

## Usage

```python
from aweai.data.profiling import DataProfiler

profiler = DataProfiler()
report = profiler.profile(X)
print(report.summary())
```

## Related Pages

- [Validation](Validation.md) — Data validation
- [Quality](Quality.md) — Data quality
