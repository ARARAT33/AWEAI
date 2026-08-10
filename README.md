# AWEAI — Universal CLI for AI/ASI/AGI Engineering

> **Copyright (c) 2026 ARARAT33 — based on AWEAI. All rights reserved.**
> Pure **terminal** (CLI-only): create, train, tune and manage AI models,
> plus hundreds of commands covering data, providers, devices, operations,
> AGI orchestration, RAG, security and an AI/ASI/AGI knowledge base.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![No Hugging Face](https://img.shields.io/badge/no-HuggingFace-red.svg)](#)

**No built-in AI model · No Hugging Face dependency · No web UI · No GUI ·
No server.** Every feature is reachable from the command line (Typer).

---

## Why AWEAI?

AWEAI is an **everything-in-one CLI** for anyone building, training or
managing AI/AGI systems from scratch:

- **Model factory** — train 16+ model architectures from zero on a light
  stack (`numpy` / optional `torch` / optional `scikit-learn`).
- **Data engineering** — scrape, crawl, import/export, clean, augment,
  synthesize, split, tokenize, embed, pipelines.
- **Providers** — OpenAI / Google / Microsoft / Anthropic / Hugging Face
  adapters (bring your own key).
- **Devices & servers** — SSH, remote hosts, cluster, distributed training,
  orchestration, hardware detection.
- **Operations** — users, roles, permissions, auth, billing, workflows,
  schedulers, cron, agents.
- **AGI tooling** — orchestration hooks, memory, reasoning,
  self-improvement (RSI) scaffolding.
- **RAG** — retrieval-augmented generation: index, search, ask.
- **AI knowledge base** — concepts, timeline, roadmap, AGI levels.
- **Bulk commands** — 300+ declarative utilities (math, string, json, file,
  net, time, crypto, ml, text, image, audio, video, db, cloud, llm, rl,
  neuro, knowledge).
- **Command registry** — `aweai commands list/search/describe/count`.
- **Wiki generator** — `aweai wiki build` generates `docs/wiki/*.md`.
- **Autotest** — one-command full-system self-check.

---

## Install

```bash
git clone https://github.com/ARARAT33/AWEAI.git
cd AWEAI
pip install -e .            # core (numpy + typer)
pip install -e ".[all]"     # + torch, onnx, scikit-learn, psutil
```

Or from the repo root, without installing:

```bash
python -m aweai version
```

## Quick start

```bash
aweai version                # v4.0.0
aweai hardware               # detect CPU/RAM/GPU
aweai types                  # list 16+ model types

# Train a model from scratch (synthetic XOR by default)
aweai train --type mlp --name m1 --data data.csv --target label
aweai train --type ngram --name lm1 --text corpus.txt --params '{"n": 3}'

# Manage the zoo
aweai models                 # list all models
aweai model info m1          # metadata + metrics
aweai eval m1 --data test.csv --target label
aweai export m1 --fmt json
aweai quantize m1 --fmt int8
aweai export-edge m1 --fmt onnx

# Data
aweai collect synthetic --kind jsonl --rows 100 --out data/synth.jsonl
aweai data split data/synth.jsonl --ratio 0.8
aweai data tokenize corpus.txt --method word
aweai data embed corpus.txt --dim 64 --out vecs.jsonl

# AI knowledge
aweai ai explain transformer
aweai ai timeline
aweai ai roadmap

# Command universe
aweai commands count         # hundreds of commands
aweai commands search rag
aweai commands describe math add

# Wiki
aweai wiki build             # generates docs/wiki/*.md
```

## Command groups

| Group | Purpose |
| --- | --- |
| `collect` | Data collection: scraping, crawling, import/export, cleaning, synthetic data |
| `data` | Data management: datasets, pipelines, preprocessing, tokenization, embedding |
| `model` | Models: train/eval/manage 16+ types, fine-tune, transfer, quantize, export |
| `providers` | API keys, external model calling, external fine-tuning (BYOK) |
| `devices` | SSH, remote hosts, cluster, distributed training, orchestration |
| `ops` | Users, roles, permissions, auth, billing, workflows, schedulers, cron, agents, AGI, RAG, security, monitoring, backup |
| `ai` | AI/ASI/AGI knowledge base (concepts, timeline, roadmap, levels, self-improve) |
| `commands` | Inspect the command universe (list/search/describe/count) |
| `wiki` | Generate the Markdown wiki |
| `math` … `knowledge` | 300+ bulk utility commands (18 groups) |

Run `aweai commands list` to see every command, or `aweai commands describe <cmd>`.

## Model types (from scratch, no HF)

| Type | Task |
| --- | --- |
| `mlp` | classification / regression |
| `cnn` | image classification (1D/2D) |
| `rnn`, `lstm`, `gru` | sequence / text |
| `transformer` | mini-Transformer (text) |
| `tft` | time-series transformer |
| `ngram` | language model |
| `gan` | generative |
| `autoencoder` | representation / anomaly |
| `vae` | generative latent |
| `diffusion` | generative (simple) |
| `vision_cnn` | image (vision) |
| `object_detection` | detection |
| `segmentation` | segmentation |
| `clustering` | unsupervised |
| `linear`, `logistic` | classical |
| `time_series` | forecasting |

## No-HuggingFace policy

AWEAI is **Hugging Face-free**: no `transformers`, no `datasets`, no `peft`.
The CI enforces this with `.github/scripts/check_hf_free.py`.

## No-UI policy (v4.0)

AWEAI is **CLI-only**. There is no web UI, no GUI, no server and no
`aweai serve` / `aweai anywhere`. Everything runs in the terminal.

## Autotest

```bash
aweai autotest          # full system self-check
aweai autotest --quick  # skip heavy smoke-trains
```

## Docs & Wiki

- `docs/` — user guide, API, architecture, changelog
- `docs/wiki/` — generated by `aweai wiki build` (hundreds of command pages)
- `wiki/` — GitHub-wiki source pages (auto-published to the repo wiki)

## License & attribution

- License: **MIT** — see [LICENSE](LICENSE).
- **Attribution required**: if you use or modify AWEAI, you MUST credit
  **ARARAT33** as co-author/author and state that the work is **based on
  AWEAI**. See [NOTICE](NOTICE) and the header comment in every source file:

```python
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
```
