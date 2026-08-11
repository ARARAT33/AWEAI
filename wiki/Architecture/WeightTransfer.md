# WeightTransfer

Weight Transfer enables transferring learned weights between different model architectures.

## Usage

```python
from aweai.architecture.weight_transfer import WeightTransfer

transfer = WeightTransfer()

# Transfer weights from MLP to Transformer
transfer.transfer(source_model="mlp", target_model="transformer")
```

## Related Pages

- [TransferLearning](../Models/TransferLearning.md) — Transfer learning
- [FineTuning](../Models/FineTuning.md) — Fine-tuning
- [Converter](Converter.md) — Architecture converter
