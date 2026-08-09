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
                         #  "autoencoder", "gan", "rnn", "lstm", "cnn", "transformer",
                         #  "gru", "ts_transformer", "vision_cnn", "object_detector",
                         #  "segmentation_net", ...]
```

## REST API (from `aweai serve`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | liveness + version |
| GET | `/api/hardware` | hardware + recommendation |
| GET | `/api/model-types` | list model types |
| GET | `/api/models` | list the model zoo |
| POST | `/api/models/train` | create+train a model |
| POST | `/api/models/eval` | evaluate a model |
| POST | `/api/models/export` | export a model |
| POST | `/api/models/delete` | delete a model |
| POST | `/api/data/load` | load dataset info |
| POST | `/api/data/augment` | augment texts |
| POST | `/api/rag/index` | index documents |
| POST | `/api/rag/ask` | ask RAG |
| POST | `/api/actions/run` | natural-language action |
| POST | `/api/autotest` | run the self-check suite |
| GET | `/api/languages` | list languages |
| GET/POST | `/api/config` | get/set config |
| POST | `/api/quantize` | quantize a model |
| POST | `/api/export/edge` | edge export |
| GET | `/api/edge/footprint` | edge footprint |
| POST | `/api/market` | marketplace |
| GET | `/api/integrations` | AI-tool integrations |
| POST | `/api/integrations/chat` | chat via provider |
| POST | `/api/terminal` | in-app terminal |
| GET | `/api/allc` | 10,000+ command catalog |
| GET | `/api/autoallc` | 10,000+ automation catalog |
| GET | `/docs` | Swagger UI |

## CLI exit codes

- `0` — success
- `1` — command failed (missing model, bad data, build error, etc.)
