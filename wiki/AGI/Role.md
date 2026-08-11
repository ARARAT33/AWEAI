# Role

Role management defines and assigns responsibilities to agents.

## Usage

```python
from aweai.agi.role import RoleManager

rm = RoleManager()
rm.define("researcher", capabilities=["search", "analyze"])
rm.assign(agent, role="researcher")
```

## Related Pages

- [CommunicationProtocol](CommunicationProtocol.md) — Communication protocol
- [MultiAgentCoordinator](MultiAgentCoordinator.md) — Multi-agent coordination
