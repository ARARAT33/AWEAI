# Catalog

Model catalog and zoo management.

## Overview

This module provides model catalog and zoo management.

## Usage

```bash
aweai train --type catalog --name model1 --data data.csv --target label
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
from aweai.models.catalog import *
model = Model(input_dim=4, output_dim=2)
model.fit(X, y)
preds = model.predict(X)
```
