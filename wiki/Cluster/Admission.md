# Admission

Admission control validates and potentially modifies requests to the cluster API server.

## Usage

```bash
# Create admission webhook
aweai cluster admission create validate-pods --url http://webhook:8080

# List admission webhooks
aweai cluster admission list
```

```python
from aweai.cluster.admission import AdmissionManager

am = AdmissionManager()
am.create_webhook("validate-pods", url="http://webhook:8080")
```

## Related Pages

- [Webhook](Webhook.md) — Webhooks
- [Mutator](Mutator.md) — Mutators
- [Validator](Validator.md) — Validators
