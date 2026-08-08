# AWEAI Architecture

AWEAI is a modular Python package: a thin core with zero-to-two dependencies,
plus optional extras for the full ML stack. Everything is importable and
testable without heavy frameworks.

## Layers

```
aweai/
├── config.py        configuration store (~/.aweai) + BYOK API keys
├── i18n.py         12-language translator (JSON assets + fallback)
├── ports.py         smart port selection: 8888 → +1
├── hardware.py     resource detection (CPU/RAM/GPU/VRAM) + score
├── utils.py        tokenizer, chunker, cosine, file helpers
├── models/         catalog, registry, selector, inference, APIs, trainer
├── rag/            RAG engine (json/chroma/faiss backends)
├── agents/         ReAct agent + tools
├── actions/        natural-language automation runner
└── ui/             FastAPI REST API + SPA frontend
```

## Core design decisions

1. **Zero heavy deps in core** — `aweai[all]` is optional; the base package
   installs with `typer` and `rich` only, and still works (fallback brain).
2. **Auto everything** — port auto-increment, model auto-selection by
   hardware, automatic fallback to API or tiny brain.
3. **Local-first** — no mandatory API keys; everything runs offline.
4. **Multilingual** — the i18n engine covers 12 languages with a tiny
   gettext-style API (`t("key")`), UI strings included.
5. **Extensible** — add tools to the agent, backends to RAG, models to the
   catalog, intents to actions.

## Data flow

### Chat
```
User → CLI/UI → LLM (auto backend) → reply
                  ├─ transformers (local, quantized)
                  ├─ API (BYOK, OpenAI-compatible)
                  └─ TinyBrain (fallback, zero deps)
```

### Training
```
JSONL data → load_texts() → mode:
  scratch  → TorchMiniGPT | TinyNgramLM → save model + vocab + metadata
  finetune → transformers + PEFT LoRA   → save adapter
  continue → load checkpoint → more epochs
```

### RAG
```
Documents → chunk_text() → tokenize() → index.json
Query     → tokenize()   → cosine similarity → top-k context → LLM answer
```

### Agent
```
Task → LLM (Thought/Action) → tool.call(args) → Observation → … → Final Answer
```

### Actions
```
Text → parse_action() → intent+params → pipeline runner → report
```

## Storage

| Path | Purpose |
|------|---------|
| `~/.aweai/config.json` | user config |
| `~/.aweai/api_keys.json` | BYOK keys (0600) |
| `~/.aweai/data/rag/index.json` | RAG index |
| `~/.aweai/models/<name>/` | trained models + metadata.json |

## Adding a new model to the catalog

Append a dict to `aweai/models/__init__.py::MODELS` with:
`id, family, params_b, context, min_ram_gb, vram_gb, quantizations, license, hf, languages, use, tier`.

The selector will pick it automatically when it fits the hardware.
