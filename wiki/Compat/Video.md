# Video

Video support enables processing video inputs.

## Usage

```python
from aweai.compat.video import VideoAdapter

video = VideoAdapter(provider="openai")
response = video.analyze(video_path="video.mp4", prompt="Describe this video")
```

## Related Pages

- [Audio](Audio.md) — Audio support
- [Vision](Vision.md) — Vision support
