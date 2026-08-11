# LongTermMemory

Long-term memory stores permanent knowledge and experiences.

## Usage

```python
from aweai.agi.memory import LongTermMemory

ltm = LongTermMemory(storage_path="./memory.db")
ltm.store("fact", "The earth orbits the sun")
fact = ltm.retrieve("fact")
```

## Related Pages

- [Memory](Memory.md) — Memory systems
- [EpisodicMemory](EpisodicMemory.md) — Episodic memory
- [Consolidation](Consolidation.md) — Memory consolidation
