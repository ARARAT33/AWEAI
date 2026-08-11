# Annotation

Annotations attach arbitrary metadata to cluster resources.

## Usage

```bash
# Add annotation
aweai cluster annotation add node1 --key description --value "GPU node"

# Remove annotation
aweai cluster annotation remove node1 --key description
```

```python
from aweai.cluster.annotation import AnnotationManager

am = AnnotationManager()
am.add("node1", key="description", value="GPU node")
am.remove("node1", key="description")
```

## Related Pages

- [Label](Label.md) — Labels
| [Selector](Selector.md) | Selectors |
