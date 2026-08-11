# Class

Classes group resources with common properties for easier management.

## Usage

```bash
# Create class
aweai cluster class create gpu-class --gpus 8 --memory 64GB

# Assign node to class
aweai cluster class assign node1 --class gpu-class
```

```python
from aweai.cluster.class_ import ClassManager

cm = ClassManager()
cm.create("gpu-class", gpus=8, memory="64GB")
cm.assign("node1", class_="gpu-class")
```

## Related Pages

| [Quota](Quota.md) | Quota |
| [Limit](Limit.md) | Limits |
