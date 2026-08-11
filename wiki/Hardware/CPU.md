# CPU

CPU management handles central processing unit resource allocation and optimization.

## Usage

```bash
# Check CPU info
aweai hardware cpu

# Configure CPU affinity
aweai hardware cpu affinity --cores 0-7
```

```python
from aweai.hardware.cpu import CPUManager

cpu = CPUManager()
cpu.set_affinity(cores=[0, 1, 2, 3])
```

## Related Pages

- [GPU](GPU.md) — GPU
- [Tier](Tier.md) — Resource tier
