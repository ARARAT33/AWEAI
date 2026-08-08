# AWEAI REST API

Base URL: `http://localhost:8888` (auto-increments on conflict).
Interactive docs: `http://localhost:8888/api/docs`.

## System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | health + version |
| GET | `/api/languages` | language codes + names |
| GET | `/api/config` | current config |
| POST | `/api/config` | update config (`{"values": {...}}`) |

## Hardware & models

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/hardware` | detected resources |
| GET | `/api/models` | catalog + installed models |
| GET | `/api/models/recommended` | best model for this machine |

## Chat

| Method | Path | Body |
|--------|------|------|
| POST | `/api/chat` | `{message, history[], model?}` |

## Training

| Method | Path | Body |
|--------|------|------|
| POST | `/api/train` | `{name, data, mode: scratch\|finetune\|continue, base_model?, epochs}` |

## RAG

| Method | Path | Body |
|--------|------|------|
| GET | `/api/rag/stats` | — |
| POST | `/api/rag/index` | `{path}` (file or dir) |
| POST | `/api/rag/ask` | `{query, top_k}` |

## Agents

| Method | Path | Body |
|--------|------|------|
| POST | `/api/agent/run` | `{task, max_steps}` |

## Actions (automation studio)

| Method | Path | Body |
|--------|------|------|
| POST | `/api/actions/run` | `{text, lang}` |

## Example

```bash
curl -s http://localhost:8888/api/health
curl -s http://localhost:8888/api/hardware
curl -s -X POST http://localhost:8888/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello"}'
curl -s -X POST http://localhost:8888/api/actions/run \
  -H 'Content-Type: application/json' \
  -d '{"text": "hardware"}'
```
