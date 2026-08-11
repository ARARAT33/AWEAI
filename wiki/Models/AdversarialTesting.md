# AdversarialTesting

Adversarial robustness testing.

## Overview

This module provides adversarial robustness testing.

## Usage

```bash
aweai train --type adversarialtesting --name model1 --data data.csv --target label
```

## Parameters

- `input_dim` — Input dimension
- `output_dim` — Output dimension
- `hidden_dim` — Hidden layer dimension
- `layers` — Number of layers
- `lr` — Learning rate
- `epochs` — Training epochs

## API

```python
from aweai.models.adversarialtesting import *
model = Model(input_dim=4, output_dim=2)
model.fit(X, y)
preds = model.predict(X)
```
