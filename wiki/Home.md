# AWEAI — Universal CLI for AI/ASI/AGI Engineering

> **Copyright (c) 2026 ARARAT33 — based on AWEAI. All rights reserved.**
> Pure **terminal** (CLI-only): no web UI, no GUI, no server.

AWEAI is an everything-in-one CLI for creating, training, tuning and
managing AI models from scratch — plus hundreds of commands covering data,
providers, devices, operations, AGI orchestration, RAG, security and an
AI/ASI/AGI knowledge base.

- **Model factory** — train 16+ model architectures from zero (numpy/torch/sklearn).
- **Data engineering** — scrape, crawl, import/export, clean, augment, synthesize.
- **Providers** — OpenAI / Google / Microsoft / Anthropic / HF adapters (BYOK).
- **Devices** — SSH, remote hosts, cluster, distributed training, orchestration.
- **Operations** — users, roles, auth, billing, workflows, schedulers, cron, agents.
- **AGI tooling** — orchestration hooks, memory, reasoning, self-improvement (RSI).
- **RAG** — retrieval-augmented generation (index, search, ask).
- **AI knowledge base** — concepts, timeline, roadmap, AGI levels.
- **Bulk commands** — 300+ declarative utilities (18 groups).
- **Command registry** — `aweai commands list/search/describe/count`.
- **Wiki generator** — `aweai wiki build` generates `docs/wiki/*.md`.
- **Autotest** — one-command full-system self-check.

## Quick start

```bash
aweai version
aweai hardware
aweai types
aweai train --type mlp --name m1 --data data.csv --target label
aweai models
aweai ai explain transformer
aweai commands count
aweai wiki build
```

See [CLI-Commands](CLI-Commands.md), [API](API.md),
[Architecture](Architecture.md), [Roadmap](Roadmap.md),
[Build-Instructions](Build-Instructions.md).
