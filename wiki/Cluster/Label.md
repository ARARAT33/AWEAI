# Label

Labels are key-value pairs attached to cluster resources for organization and selection.

## Usage

```bash
# Add label
aweai cluster label add node1 --key env --value production

# Remove label
aweai cluster label remove node1 --key env
```

```python
from aweai.cluster.label import LabelManager

lm = LabelManager()
lm.add("node1", key="env", value="production")
lm.remove("node1", key="env")
```

## Related Pages

- [Annotation](Annotation.md) — Annotations
| [Selector](Selector.md) | Selectors |
