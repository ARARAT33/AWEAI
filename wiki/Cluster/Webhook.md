# Webhook

Webhooks enable external HTTP callbacks for cluster events and validations.

## Usage

```bash
# Register webhook
aweai cluster webhook register my-webhook --url http://service:8080/webhook \
  --events pod-created,pod-deleted

# List webhooks
aweai cluster webhook list
```

```python
from aweai.cluster.webhook import WebhookManager

wm = WebhookManager()
wm.register("my-webhook", url="http://service:8080/webhook",
            events=["pod-created", "pod-deleted"])
```

## Related Pages

- [Admission](Admission.md) — Admission
- [Mutator](Mutator.md) — Mutators
- [Validator](Validator.md) — Validators
