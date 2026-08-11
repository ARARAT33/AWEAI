# Autoencoder

Autoencoder for representation learning and anomaly detection.

## Overview

This module provides autoencoder for representation learning and anomaly detection.

## Usage

```bash
aweai train --type autoencoder --name model1 --data data.csv --target label
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
from aweai.models.autoencoder import *
model = Model(input_dim=4, output_dim=2)
model.fit(X, y)
preds = model.predict(X)
```
