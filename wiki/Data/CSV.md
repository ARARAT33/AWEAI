# CSV

CSV (Comma-Separated Values) format support.

## Usage

```python
from aweai.data.formats import CSVFormat

csv = CSVFormat()
data = csv.read("data.csv")
csv.write(data, "output.csv")
```

## Related Pages

- [Formats](Formats.md) — Data formats
- [Loaders](Loaders.md) — Data loaders
