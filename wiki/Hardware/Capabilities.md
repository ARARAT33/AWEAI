# Capabilities

Hardware capabilities define what operations and features are supported.

## Usage

```python
from aweai.hardware.capabilities import get_capabilities

caps = get_capabilities()
print(caps.supports_fp16)
print(caps.supports_tensor_cores)
```

## Related Pages

- [Detection](Detection.md) — Hardware detection
- [Tier](Tier.md) — Resource tier
