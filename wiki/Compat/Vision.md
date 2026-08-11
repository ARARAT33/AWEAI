# Vision

Vision support enables processing images and visual inputs.

## Usage

```python
from aweai.compat.vision import VisionAdapter

vision = VisionAdapter(provider="openai")
response = vision.analyze(image_path="image.jpg", prompt="Describe this image")
```

## Related Pages

- [ChatCompletions](ChatCompletions.md) — Chat completions
- [Vision](../Models/Vision.md) — Vision models
