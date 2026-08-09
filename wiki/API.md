# API

## Python API

```python
from aweai.train import train
from aweai.models import list_model_types
from aweai.utils import tokenize, chunk_text, cosine_similarity, safe_filename
from aweai.i18n import LANGUAGES, t

# Train and save a model -> dict {name, version, model_type, path}
res = train("mlp", "my_model", X=[[0,0],[1,1]], y=[0,1], params={"epochs": 10})

# Train without saving -> live model object with .predict() / .labels_
live = train("kmeans", "live", X=data, params={"k": 3}, save=False)
live.predict(X)          # or live.labels_

list_model_types()       # ["mlp", "linear", "logistic", "kmeans", "ngram",
                         #  "autoencoder", "gan", "rnn", "lstm", "cnn", "transformer", ...]
```

## REST API (from `aweai serve`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | liveness + version |
| GET | `/api/models` | list the model zoo |
| POST | `/api/train` | start a training job |
| GET | `/api/terminal/ws` | in-app terminal WebSocket |
| GET | `/api/autotest` | run the self-check suite |

## CLI exit codes

- `0` — success
- `1` — command failed (missing model, bad data, build error, etc.)
