# NPU

NPU management handles Neural Processing Unit resource allocation and optimization.

## Usage

```bash
# Check NPU info
aweai hardware npu

# Configure NPU
aweai hardware npu configure --cores 4
```

```python
from aweai.hardware.npu import NPUManager

npu = NPUManager()
npu.configure(cores=4)
```

## Related Pages

- [GPU](GPU.md) — GPU
- [TPU](TPU.md) — TPU
- [Tier](Tier.md) — Resource tier
