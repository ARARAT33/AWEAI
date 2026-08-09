# UI Guide

Launch the browser UI with:

```bash
aweai serve            # http://localhost:8888
aweai serve --port 9000 --host 0.0.0.0
```

## Huge menu system

The sidebar is a combinatorial menu system — 22 groups × sub-actions × variants produce **100,000+ navigable pages/menus**, all backed by the factory's REST API. Fully responsive: works on mobile, tablet and desktop.

- **Dashboard** — hardware, recommendations, live loss curves.
- **Wizard · Train** — train / continue / tune / recommend.
- **Model Zoo** — list / export / import / delete / compare.
- **Datasets** — load / split / augment / tokenize / normalize.
- **Hyperparameters** — grid / random / bayes / defaults / history.
- **Evaluation** — report / curves / confusion / compare / metrics.
- **Quantization** — float16 / int8 / uint8 / int4.
- **Edge Export** — onnx / tflite / torchscript / edge_json / footprint.
- **Distributed** — dtrain / dworld / workers / nodes / backend.
- **RAG** — index / ask / stats / clear / documents.
- **Marketplace** — list / search / publish / download / rate / stats.
- **AI Tools** — BYOK integrations (OpenAI / Gemini / Azure / Claude / HF).
- **Terminal** — full in-app terminal page.
- **Megamenus (allc)** — browse the 10,000+ command catalog.
- **Automations** — NL actions + pipelines.
- **Debuggers** — model inspection, edge footprint.
- **Libraries** — inventory of every library.
- **Tests** — run the test suite.
- **Autotest** — full system check.
- **Config / i18n** — 12 languages.
- **API / Docs** — Swagger at `/docs`.

## In-app terminal

The in-app terminal is a real REPL exposed over the web API. Everything available from the CLI is available there. Press **Ctrl+`** anywhere to toggle the terminal drawer.

```text
> aweai allc --category train
> aweai autotest --quick
> aweai train --type mlp --name demo --params '{"epochs": 5}'
> market list
```

## Global search

The sidebar search box queries the 10,000+ command catalog live and jumps straight to the Megamenus page.

## REST API

The UI is backed by a REST API on the same port (Swagger at `/docs`):

- `GET /api/health`, `GET /api/hardware`, `GET /api/model-types`, `GET /api/models`
- `POST /api/models/train`, `POST /api/models/eval`, `POST /api/models/export`, `POST /api/models/delete`
- `POST /api/data/load`, `POST /api/data/augment`
- `POST /api/rag/index`, `POST /api/rag/ask`
- `POST /api/actions/run`, `POST /api/autotest`
- `GET /api/languages`, `GET/POST /api/config`
- `POST /api/quantize`, `POST /api/export/edge`, `GET /api/edge/footprint`
- `POST /api/market`, `GET /api/integrations`, `POST /api/integrations/chat`
- `POST /api/terminal`, `GET /api/allc`, `GET /api/autoallc`
