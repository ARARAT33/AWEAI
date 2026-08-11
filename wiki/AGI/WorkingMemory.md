# WorkingMemory

Working memory holds temporary information during reasoning and problem-solving.

## Usage

```python
from aweai.agi.memory import WorkingMemory

wm = WorkingMemory(capacity=7)  # Miller's magic number
wm.store("current_task", "Analyze data")
wm.update("current_step", 3)
```

## Related Pages

- [Memory](Memory.md) — Memory systems
- [LongTermMemory](LongTermMemory.md) — Long-term memory
