# ProceduralMemory

Procedural memory stores skills and procedures for execution.

## Usage

```python
from aweai.agi.memory import ProceduralMemory

pm = ProceduralMemory()
pm.store("train_model", steps=["load data", "preprocess", "train", "evaluate"])
pm.execute("train_model")
```

## Related Pages

- [Memory](Memory.md) — Memory systems
- [Consolidation](Consolidation.md) — Memory consolidation
