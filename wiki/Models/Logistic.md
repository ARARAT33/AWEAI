# Logistic

Logistic regression classifier.

## Overview

This module provides logistic regression classifier.

## Usage

```bash
aweai train --type logistic --name model1 --data data.csv --target label
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
from aweai.models.logistic import *
model = Model(input_dim=4, output_dim=2)
model.fit(X, y)
preds = model.predict(X)
```
