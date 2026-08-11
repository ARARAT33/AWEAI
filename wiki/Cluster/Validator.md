# Validator

Validators validate cluster objects during admission control.

## Usage

```bash
# Create validator
aweai cluster validator create require-gpu --rule "Must have GPU request"

# List validators
aweai cluster validator list
```

```python
from aweai.cluster.validator import ValidatorManager

vm = ValidatorManager()
vm.create("require-gpu", rule="Must have GPU request")
```

## Related Pages

- [Admission](Admission.md) — Admission
- [Webhook](Webhook.md) — Webhooks
- [Mutator](Mutator.md) — Mutators
