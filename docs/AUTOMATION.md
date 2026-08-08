# Automation

AWEAI automation is built on **natural-language actions** and **pipelines**.

## Actions (`aweai.actions`)

```python
from aweai.actions import parse_action, run_action
parse_action("train an mlp model named demo")   # {"action": "train", "kwargs": {...}}
run_action("list all models")
run_action("export the model demo to json")
```

## Pipelines

```python
from aweai.actions import save_pipeline, run_pipeline
steps = [{"action": "train", "kwargs": {"model_type": "mlp", "name": "auto_1"}}]
save_pipeline("p1", steps)
run_pipeline("p1")
```

## Batch jobs

```python
from aweai.actions import run_batch
run_batch(["list all models", "recommend a model for regression"])
```
