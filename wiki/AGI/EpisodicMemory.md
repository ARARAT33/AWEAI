# EpisodicMemory

Episodic memory stores personal experiences and events.

## Usage

```python
from aweai.agi.memory import EpisodicMemory

em = EpisodicMemory()
em.store(event="Met John at conference", timestamp="2024-01-15", emotions=["happy"])
memories = em.retrieve(query="conference")
```

## Related Pages

- [Memory](Memory.md) — Memory systems
- [AutobiographicalMemory](AutobiographicalMemory.md) — Autobiographical memory
- [LongTermMemory](LongTermMemory.md) — Long-term memory
