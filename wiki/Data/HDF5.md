# HDF5

HDF5 (Hierarchical Data Format) support.

## Usage

```python
from aweai.data.formats import HDF5Format

hdf5 = HDF5Format()
data = hdf5.read("data.h5")
hdf5.write(data, "output.h5")
```

## Related Pages

- [Feather](Feather.md) — Feather format
- [Formats](Formats.md) — Data formats
