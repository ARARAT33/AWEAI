# ResourceDetection

Resource detection automatically identifies available hardware resources for optimal training configuration.

## Usage

```bash
# Detect available resources
aweai hardware detect

# Show resource recommendations
aweai hardware recommend --task train --model-type transformer
```

```python
from aweai.scale.resource_detection import detect_resources

resources = detect_resources()
print(resources.gpus)
print(resources.memory)
print(resources.storage)
```

## Detected Resources

| Resource | Description |
|----------|-------------|
| GPUs | Number and type of GPUs |
| TPUs | TPU availability |
| Memory | GPU and system RAM |
| Storage | SSD/HDD capacity and speed |
| Network | Interconnect type and bandwidth |

## Related Pages

- [Hardware](../Hardware/Detection.md) — Hardware detection
- [UnifiedTrainer](UnifiedTrainer.md) — Unified trainer
