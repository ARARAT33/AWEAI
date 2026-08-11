# YAML

YAML format support.

## Usage

```python
from aweai.data.formats import YAMLFormat

yaml = YAMLFormat()
data = yaml.read("data.yaml")
yaml.write(data, "output.yaml")
```

## Related Pages

- [XML](XML.md) — XML format
- [TOML](TOML.md) — TOML format
- [Formats](Formats.md) — Data formats
