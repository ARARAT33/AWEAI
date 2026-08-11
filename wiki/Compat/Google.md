# Google

Google AI-compatible layer.

## Overview

Google AI-compatible layer.

## Usage

```bash
aweai compat start --port 8000
aweai compat route openai gpt-4 --fallback local-model
```

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
