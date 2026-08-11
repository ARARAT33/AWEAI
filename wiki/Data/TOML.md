# TOML

TOML format support.

## Usage

```python
from aweai.data.formats import TOMLFormat

toml = TOMLFormat()
data = toml.read("data.toml")
toml.write(data, "output.toml")
```

## Related Pages

- [YAML](YAML.md) — YAML format
- [Formats](Formats.md) — Data formats
