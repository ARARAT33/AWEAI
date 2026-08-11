# MLP

Multi-Layer Perceptron for classification and regression.

## Overview

This module provides multi-layer perceptron for classification and regression.

## Usage

```bash
aweai train --type mlp --name model1 --data data.csv --target label
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
from aweai.models.mlp import *
model = Model(input_dim=4, output_dim=2)
model.fit(X, y)
preds = model.predict(X)
```
