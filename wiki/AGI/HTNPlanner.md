# HTNPlanner

Hierarchical Task Network (HTN) planner decomposes complex tasks into hierarchical subtasks.

## Usage

```python
from aweai.agi.planning import HTNPlanner

planner = HTNPlanner()
plan = planner.plan("Build web app", domain_knowledge=...)
```

## Related Pages

- [Planning](Planning.md) — Planning
- [ConstraintSatisfier](ConstraintSatisfier.md) — Constraint satisfaction
