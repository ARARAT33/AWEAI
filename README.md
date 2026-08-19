# AWEAI — Universal CLI for AI/ASI/AGI Engineering

> **Copyright (c) 2026 ARARAT33 — based on AWEAI. All rights reserved.**
> Pure **terminal** (CLI-only): create, train, tune and manage AI models,
> plus hundreds of commands covering data, providers, devices, operations,
> AGI orchestration, RAG, security, multi-layer watermarking, and an AI/ASI/AGI knowledge base.

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
- **Indelible Multi-Layer Watermarking** — multi-layer visible headers, zero-width unicode steganography (ZWC), HMAC cryptographic signatures, floating-point array perturbation, and tamper detection across models, datasets, files, and exported artifacts.
- **Data engineering** — scrape, crawl, import/export, clean, augment, synthesize, split, tokenize, embed, pipelines.
- **Providers** — normalized adapters and routing for major AI ecosystems (bring your own key).
- **Universal AI Ecosystem Gateway** — one AWEAI control plane for provider discovery, capability routing, policy, audit and adapter contracts.
- **Universal AI Company Tooling & Product Engine** — capability registry, secure vault, artifact lineage ledger, telemetry, and model governance audit.
- **Devices & servers** — SSH, remote hosts, cluster, distributed training, orchestration, hardware detection.
- **Operations** — users, roles, permissions, auth, billing, workflows, schedulers, cron, agents.
- **AGI tooling** — orchestration hooks, memory, reasoning, self-improvement (RSI) scaffolding.
- **RAG** — retrieval-augmented generation: index, search, ask.
- **AI knowledge base** — concepts, timeline, roadmap, AGI levels.
- **Bulk commands** — 300+ declarative utilities (math, string, json, file, net, time, crypto, ml, text, image, audio, video, db, cloud, llm, rl, neuro, knowledge).
- **Command registry** — `aweai commands list/search/describe/count`.
- **Wiki generator** — `aweai wiki build` generates watermarked `docs/wiki/*.md`.
- **Autotest** — one-command full-system self-check (including watermark verification).

---

## Indelible Multi-Layer Watermarking & Steganography

AWEAI includes an advanced, multi-layered watermarking engine (`aweai/watermark.py` and `aweai watermark` CLI):

1. **Visible Watermark Headers & Footers**: Prominent copyright metadata (`Copyright (c) 2026 ARARAT33`).
2. **Invisible Zero-Width Unicode Steganography (ZWC)**: Concealed cryptographic signatures embedded directly into text strings, Markdown docs, and JSON fields without affecting visual appearance.
3. **HMAC Cryptographic Signatures**: Hash-based integrity checks preventing payload tampering.
4. **Floating-Point Array Perturbation**: Micro-perturbations embedded in model weights and float tensors.
5. **Tamper Detection**: Verification detects any unauthorized modification or stripping of watermarks.

### Watermark CLI Commands

```bash
aweai watermark status
aweai watermark embed "Sample dataset text or json file"
aweai watermark verify "Sample text or json file"
aweai watermark extract "Sample text or json file"
aweai watermark inspect "my_model.json"
```

---

## Universal AI Ecosystem Gateway

AWEAI provides a normalized ecosystem layer covering providers such as
OpenAI, Anthropic, Google Gemini, Microsoft Azure AI, Meta, xAI, Mistral,
Cohere, DeepSeek, Qwen, Zhipu, Moonshot, MiniMax, Groq, Together, Fireworks,
Perplexity, OpenRouter, Hugging Face, Replicate, Stability AI, ElevenLabs,
AssemblyAI, Ollama and LM Studio.

```bash
aweai tools list --category ecosystem
aweai tools describe --name ecosystem_route
aweai tools run --name ecosystem_route --params '{"capability":"reasoning"}'
aweai tools run --name ecosystem_policy --params '{"action":"enforce"}'
aweai tools run --name ecosystem_audit
aweai tools run --name ecosystem_contract
```

---

## Universal AI Company Tooling

`aweai/company.py` and `aweai/product.py` define the engineering and product control surface: capability contracts, secure vault encryption, watermarked artifact ledgers, telemetry, and governance audits.

```python
from aweai.company import CompanyToolRegistry
from aweai.product import WatermarkedArtifactRegistry, AWEAISecureVault

registry = CompanyToolRegistry()
print(registry.manifest())

vault = AWEAISecureVault(owner="ARARAT33")
sealed = vault.seal({"config": "prod_setting"})
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

aweai watermark verify m1
aweai watermark status

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
aweai autotest
```

## Command groups

| Group | Purpose |
| --- | --- |
| `collect` | Data collection: scraping, crawling, import/export, cleaning, synthetic data |
| `data` | Data management: datasets, pipelines, preprocessing, tokenization, embedding |
| `model` | Models: train/eval/manage 16+ types, fine-tune, transfer, quantize, export |
| `watermark` | Multi-layer indelible watermarking & steganography |
| `providers` | API keys, external model calling, external fine-tuning (BYOK) |
| `devices` | SSH, remote hosts, cluster, distributed training |
| `ops` | Users, roles, permissions, auth, billing, workflows, schedulers, cron, agents, AGI, RAG, security |
| `ai` | AI/ASI/AGI knowledge base (concepts, timeline, roadmap, levels, self-improve) |
| `commands` | Inspect the command universe (list/search/describe/count) |
| `wiki` | Generate the Markdown wiki |
| `tools` | Universal registered tool registry |
| `math` … `knowledge` | 300+ bulk utility commands (18 groups) |

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

AWEAI is **CLI-only**. There is no web UI, no GUI, no server. Everything runs in the terminal.

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