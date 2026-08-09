# UI Guide

Launch the browser UI with:

```bash
aweai serve            # http://localhost:8888
aweai serve --port 9000 --host 0.0.0.0
```

## What's in the UI

- **Dashboard** — system status, hardware, quick actions.
- **Model zoo** — list, create, train, evaluate, export, quantize, delete models.
- **Training** — configure and launch training jobs (single or distributed).
- **Marketplace** — browse, publish, download, rate models.
- **Terminal** — full in-app terminal: every CLI command runs inside the browser (`aweai terminal` in the UI too).
- **AI tools** — BYOK integration chat (OpenAI / Gemini / Azure / Claude / HF).
- **Debuggers** — model inspection, dataset preview, autotest runner.
- **Libraries & tests** — on-demand library list, autotest report, export status.

## In-app terminal

The in-app terminal is a real PTY-backed REPL exposed over the web API. Everything available from the CLI is available there:

```text
> aweai allc --category train
> aweai autotest --quick
> aweai train --type mlp --name demo --params '{"epochs": 5}'
```

## REST API

The UI is backed by a REST API on the same port:

- `GET /api/health`
- `GET /api/models`
- `POST /api/train`
- `GET /api/terminal/ws` — terminal WebSocket
- `GET /api/autotest`
