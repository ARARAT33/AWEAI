# CapabilityMetric

Capability metrics measure and track agent capabilities over time.

## Usage

```python
from aweai.agi.metrics import CapabilityMetric

cm = CapabilityMetric()
score = cm.evaluate(agent, task)
history = cm.track(agent, time_period="30d")
```

## Related Pages

- [Agent](Agent.md) — Agent framework
- [RSI](RSI.md) — Recursive self-improvement
