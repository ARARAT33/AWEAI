# XML

XML (Extensible Markup Language) format support.

## Usage

```python
from aweai.data.formats import XMLFormat

xml = XMLFormat()
data = xml.read("data.xml")
xml.write(data, "output.xml")
```

## Related Pages

- [JSON](JSON.md) — JSON format
- [Formats](Formats.md) — Data formats
