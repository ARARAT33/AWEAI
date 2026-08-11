# Lineage

Data lineage tracks data flow and transformations through pipelines.

## Usage

```python
from aweai.data.lineage import LineageTracker

tracker = LineageTracker()
tracker.add_step("load", "raw_data.csv")
tracker.add_step("transform", "cleaned_data.csv")
tracker.visualize()
```

## Related Pages

- [Pipelines](Pipelines.md) — Data pipelines
- [Versioning](Versioning.md) — Data versioning
