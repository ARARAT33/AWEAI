# IdentityPersistence

Identity persistence maintains agent continuity across sessions and restarts.

## Usage

```python
from aweai.agi.identity import IdentityPersistence

ip = IdentityPersistence(storage_path="./identity.json")
ip.save(agent_state)
restored_agent = ip.load()
```

## Related Pages

- [SelfModel](SelfModel.md) — Self-model
- [AutobiographicalMemory](AutobiographicalMemory.md) — Autobiographical memory
