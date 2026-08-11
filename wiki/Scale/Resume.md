# Resume

Resume allows training to continue from a saved checkpoint after interruption or for continued training.

## Usage

```bash
# Resume training
aweai continue-train my_model --checkpoint ./checkpoints/epoch_10.pt \
  --data data.csv --epochs 20
```

```python
from aweai.scale.resume import ResumeTrainer

trainer = ResumeTrainer(
    model_type="transformer",
    checkpoint_path="./checkpoints/epoch_10.pt"
)

trainer.fit(X_train, y_train, epochs=20)
```

## Related Pages

- [Checkpoint](Checkpoint.md) — Checkpointing
- [ElasticTraining](ElasticTraining.md) — Elastic training
