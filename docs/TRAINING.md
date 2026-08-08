# Training

`aweai.train.train()` trains a model **from scratch**:

```python
from aweai.train import train
res = train("mlp", "m1", X=X, y=y, params={"epochs": 50, "hidden": [16, 8]})
```

- **Continue / fine-tune**: `aweai.train.continue_training(name, data_path=..., epochs=...)`
- **Hyperparameter tuning**: `aweai.train.grid_search`, `random_search`, `tune`
- **Early stopping** and **val metrics** are supported by the trainer
- **Resource-adaptive defaults**: `aweai.selector.recommend(task)` picks the
  best model type and hyperparameters for the current hardware
