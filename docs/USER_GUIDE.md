# User Guide

## Quick start

```bash
pip install -e .
aweai autotest        # verify the whole system
aweai serve           # browser UI on http://localhost:8888
```

## Train your first model

```bash
aweai train --type mlp --name my_model --data data.csv --target label --params '{"epochs": 50}'
```

or in Python:

```python
from aweai.train import train
res = train("mlp", "my_model", X=[[0,0],[1,1]], y=[0,1], params={"epochs": 10})
```

## CLI overview

- `aweai hardware` — show detected hardware and resource tier
- `aweai recommend --task classification` — best model type for this machine
- `aweai train/continue/eval/models/export/delete/compare` — model lifecycle
- `aweai data load/split/augment` — data tools
- `aweai rag index/ask` — RAG
- `aweai actions "..."` — natural-language automation
- `aweai autotest` — full system self-check
- `aweai serve` — launch the UI
