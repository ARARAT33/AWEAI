# Distributed Training (v3.0)

Multi-GPU / multi-node / multi-thread training for AWEAI models.

```bash
aweai dworld                                   # detect GPUs, nodes, backend
aweai dtrain mlp --name d1 --data train.csv --workers 4 --backend auto
```

- `backend=auto` picks `torch` when torch + GPUs are available, otherwise a
  built-in multi-thread data-parallel backend that works with pure-numpy
  models and zero extra dependencies.
- `workers` defaults to detected GPU count (or CPUs when no GPU).

## Python API

```python
from aweai.distributed import train_distributed, detect_world

world = detect_world()          # {'gpus':..., 'cpus':..., 'nodes':..., 'backend':...}
res = train_distributed("mlp", "d1", X, y=y, workers=4, epochs=30)
```

The engine averages worker state dicts (recursive, list-aware) so results
are deterministic and reproducible on a single machine.
