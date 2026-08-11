# AutobiographicalMemory

Autobiographical memory stores the agent's life history and self-narrative.

## Usage

```python
from aweai.agi.memory import AutobiographicalMemory

am = AutobiographicalMemory()
am.store(event="First training run", date="2024-01-01", outcome="success")
narrative = am.generate_narrative()
```

## Related Pages

- [EpisodicMemory](EpisodicMemory.md) — Episodic memory
- [Memory](Memory.md) — Memory systems
