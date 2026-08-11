# Mutator

Mutators modify cluster objects during admission control.

## Usage

```bash
# Create mutator
aweai cluster mutator create add-labels --action add-labels --labels team=ml

# List mutators
aweai cluster mutator list
```

```python
from aweai.cluster.mutator import MutatorManager

mm = MutatorManager()
mm.create("add-labels", action="add-labels", labels={"team": "ml"})
```

## Related Pages

- [Admission](Admission.md) — Admission
- [Webhook](Webhook.md) — Webhooks
- [Validator](Validator.md) — Validators
