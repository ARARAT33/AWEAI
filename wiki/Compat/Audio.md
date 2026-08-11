# Audio

Audio support enables processing audio inputs and outputs.

## Usage

```python
from aweai.compat.audio import AudioAdapter

audio = AudioAdapter(provider="openai")
response = audio.transcribe(audio_path="audio.mp3")
```

## Related Pages

- [Vision](Vision.md) — Vision support
- [ChatCompletions](ChatCompletions.md) — Chat completions
