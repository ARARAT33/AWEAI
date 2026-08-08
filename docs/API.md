# REST API

Base URL: `http://localhost:8888` (auto +1 if busy). Interactive docs at
`/docs` (OpenAPI).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | health check |
| GET | `/api/hardware` | hardware + recommendation |
| GET | `/api/model-types` | model zoo types |
| GET | `/api/models` | list models |
| POST | `/api/models/train` | create + train a model |
| POST | `/api/models/eval` | evaluate a model |
| POST | `/api/models/export` | export a model |
| POST | `/api/models/delete` | delete a model |
| POST | `/api/data/load` | load dataset info |
| POST | `/api/data/augment` | augment texts |
| POST | `/api/rag/index` | index documents |
| POST | `/api/rag/ask` | ask RAG |
| POST | `/api/actions/run` | run natural-language action |
| POST | `/api/autotest` | run autotest |
| GET | `/api/languages` | list languages |
| GET/POST | `/api/config` | get/set config |
