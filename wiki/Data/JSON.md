# JSON

JSON (JavaScript Object Notation) format support.

## Usage

```python
from aweai.data.formats import JSONFormat

json = JSONFormat()
data = json.read("data.json")
json.write(data, "output.json")
```

## Related Pages

- [CSV](CSV.md) — CSV format
- [Formats](Formats.md) — Data formats
