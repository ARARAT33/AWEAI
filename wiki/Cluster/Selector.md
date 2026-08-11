# Selector

Selectors enable filtering and targeting of cluster resources based on labels and fields.

## Usage

```bash
# Select nodes
aweai cluster selector nodes --selector "env=production,gpu=true"
```

```python
from aweai.cluster.selector import SelectorManager

sm = SelectorManager()
nodes = sm.select_nodes(selector="env=production,gpu=true")
```

## Related Pages

- [Label](Label.md) — Labels
| [Annotation](Annotation.md) | Annotations |
| [Placement](Placement.md) | Placement |
