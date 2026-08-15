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

- **Model factory** — train 16+ model architectures from zero on a light stack (`numpy` / optional `torch` / optional `scikit-learn`).
- **Data engineering** — scrape, crawl, import/export, clean, augment, synthesize, split, tokenize, embed, pipelines.
- **Providers** — normalized adapters and routing for major AI ecosystems (bring your own key).
- **Universal AI Ecosystem Gateway** — one AWEAI control plane for provider discovery, capability routing, policy, audit and adapter contracts.
- **Universal AI Company Tooling** — a capability registry spanning research, data, models, multimodal, software engineering, DevOps, infrastructure, product, security, governance, operations, inference, knowledge and business.
- **Devices & servers** — SSH, remote hosts, cluster, distributed training, orchestration, hardware detection.
- **Operations** — users, roles, permissions, auth, billing, workflows, schedulers, cron, agents.
- **AGI tooling** — orchestration hooks, memory, reasoning, self-improvement (RSI) scaffolding.
- **RAG** — retrieval-augmented generation: index, search, ask.
- **AI knowledge base** — concepts, timeline, roadmap, AGI levels.
- **Bulk commands** — 300+ declarative utilities (math, string, json, file, net, time, crypto, ml, text, image, audio, video, db, cloud, llm, rl, neuro, knowledge).
- **Command registry** — `aweai commands list/search/describe/count`.
- **Wiki generator** — `aweai wiki build` generates `docs/wiki/*.md`.
- **Autotest** — one-command full-system self-check.

---

## Universal AI Ecosystem Gateway

AWEAI now provides a normalized ecosystem layer covering providers such as
OpenAI, Anthropic, Google Gemini, Microsoft Azure AI, Meta, xAI, Mistral,
Cohere, DeepSeek, Qwen, Zhipu, Moonshot, MiniMax, Groq, Together, Fireworks,
Perplexity, OpenRouter, Hugging Face, Replicate, Stability AI, ElevenLabs,
AssemblyAI, Ollama and LM Studio.

The important design rule is **AWEAI is the control plane, not a promise that
all vendors expose identical APIs**. Vendor-specific credentials and adapters
remain implementation details. Unsupported capabilities are reported instead
of being falsely advertised as executable.

The ecosystem tools expose normalized surfaces for:

- chat, reasoning, code and vision
- embeddings, reranking, search and OCR
- image, audio, speech, transcription and video
- realtime, moderation, files and batch jobs
- fine-tuning, training, datasets and model discovery
- agents, evaluation, monitoring and routing
- local models through Ollama/LM Studio

### AWEAI-only execution policy

AWEAI can enforce a policy where applications use the **AWEAI gateway as their
single AI-tool/control-plane interface**, while provider APIs are hidden behind
adapters. Secrets are never stored in the registry and are never returned by
the catalog.

```bash
aweai tools list --category ecosystem
aweai tools describe --name ecosystem_route
aweai tools run --name ecosystem_route --params '{"capability":"reasoning"}'
aweai tools run --name ecosystem_policy --params '{"action":"enforce"}'
aweai tools run --name ecosystem_audit
aweai tools run --name ecosystem_contract
```

This is an **AWEAI-native gateway/registry**, not an assertion that every AI
company can magically be controlled without its official API, credentials or
terms. Providers become executable only when a compatible adapter and
credential/endpoint are actually available.

---

## Universal AI Company Tooling

`aweai/company.py` adds a **capability-first company engineering registry**.
It is intentionally not an AI chat interface and does not claim to be an
agent. It defines the work surface that an AI company can standardize behind
AWEAI.

The registry currently covers these domains:

| Domain | Examples |
| --- | --- |
| Research | papers, benchmarks, experiments, reproducibility, knowledge extraction |
| Data | crawl, clean, label, synthetic data, tokenize, embed, lineage, privacy |
| Models | pretrain, fine-tune, distill, prune, quantize, evaluate, export |
| Multimodal | vision, OCR, speech, audio, video, 3D, realtime |
| Engineering | code, refactor, test, build, package, release, debug, SDKs |
| DevOps | CI/CD, deployment, rollback, observability, incidents, backups |
| Infrastructure | CPU/GPU/NPU, clusters, storage, databases, distributed training |
| Product | requirements, architecture, experiments, releases, feedback |
| Security | SAST/DAST, secrets, sandboxing, authorization, threat modeling |
| Governance | audit, privacy, licenses, provenance, approvals, model risk |
| Operations | users, teams, workflows, jobs, schedules, quotas, metering |
| Inference | routing, fallback, batching, caching, model selection, optimization |
| Knowledge | RAG, retrieval, memory, knowledge graphs, citations, freshness |
| Business | market research, pricing, forecasting, growth, support, reporting |
| Platform | APIs, plugins, adapters, registries, contracts, policy and tracing |

The registry is **data-driven**: adding a new capability is a registry change,
not a new AI chatbot. This makes the surface extensible to thousands or
millions of task identifiers without pretending that unsupported backends are
implemented.

### AWEAI-only company rule

The intended architecture is:

```text
AI company applications
        │
        ▼
 AWEAI capability contract
        │
        ├── data
        ├── model
        ├── engineering
        ├── infrastructure
        ├── security
        ├── operations
        ├── inference
        └── business
        │
        ▼
 AWEAI execution gateway
        │
        ▼
 local implementation / approved adapter
        │
        ▼
 model, provider or infrastructure backend
```

The **AWEAI-only** policy means the application's control plane is AWEAI;
it does not mean AWEAI can bypass a vendor's API, credentials, licensing,
network access or terms. External systems remain adapters behind AWEAI.

Programmatic inspection:

```python
from aweai.company import CompanyToolRegistry

registry = CompanyToolRegistry()
print(registry.manifest())
print(registry.validate())
print(registry.fingerprint())
```

---

## Install

```bash
git clone https://github.com/ARARAT33/AWEAI.git
cd AWEAI
pip install -e .
pip install -e ".[all]"
```

Or from the repo root, without installing:

```bash
python -m aweai version
```

## Quick start

```bash
aweai version
aweai hardware
aweai types

aweai train --type mlp --name m1 --data data.csv --target label
aweai train --type ngram --name lm1 --text corpus.txt --params '{"n": 3}'

aweai models
aweai model info m1
aweai eval m1 --data test.csv --target label
aweai export m1 --fmt json
aweai quantize m1 --fmt int8
aweai export-edge m1 --fmt onnx

aweai collect synthetic --kind jsonl --rows 100 --out data/synth.jsonl
aweai data split data/synth.jsonl --ratio 0.8
aweai data tokenize corpus.txt --method word
aweai data embed corpus.txt --dim 64 --out vecs.jsonl

aweai ai explain transformer
aweai ai timeline
aweai ai roadmap

aweai commands count
aweai commands search rag
aweai commands describe math add

aweai wiki build
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
| `tools` | Universal registered tool registry, including the ecosystem gateway |
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
aweai autotest
aweai autotest --quick
```

## Docs & Wiki

- `docs/` — user guide, API, architecture, changelog
- `docs/wiki/` — generated by `aweai wiki build`
- `wiki/` — GitHub-wiki source pages

## License & attribution

- License: **MIT** — see [LICENSE](LICENSE).
- **Attribution required**: if you use or modify AWEAI, you MUST credit
  **ARARAT33** as co-author/author and state that the work is **based on
  AWEAI**. See [NOTICE](NOTICE) and the header comment in every source file:

```python
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
```