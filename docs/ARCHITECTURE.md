# AWEAI Architecture

AWEAI v2.0 is a modular, layered Python application. It is designed to be
lightweight at the core and progressively heavier as optional extras are
installed — the same code runs on a phone and on a multi-GPU server.

## Layers

```
┌────────────────────────────────────────────────────────────┐
│  UI layer:  FastAPI (aweai/ui)  +  SPA (static/)  +  CLI   │
├────────────────────────────────────────────────────────────┤
│  Domain:    actions → agents → rag → models → trainer      │
├────────────────────────────────────────────────────────────┤
│  Platform:  config · i18n · hardware · ports · utils       │
└────────────────────────────────────────────────────────────┘
```

## Modules

### `aweai/config.py`
- `Config` — JSON-backed settings in `~/.aweai/config.json` (or `$AWEAI_HOME`).
- `ApiKeyStore` — API keys in `~/.aweai/api_keys.json` with `0600` mode.
- `ensure_runtime_dirs()` — creates `models/`, `rag/`, `actions/`, `logs/`.

### `aweai/ports.py`
- `resolve_port(preferred=8888)` — checks TCP availability, returns the first
  free port starting at the preferred value, incrementing by 1 on conflict.

### `aweai/hardware.py`
- Zero heavy deps: stdlib + optional `psutil`/`torch`.
- Detects CPU cores/freq, RAM total/free, NVIDIA GPUs (via `torch.cuda` or
  `nvidia-smi`), Apple MPS, disk space, Android.
- Computes a rough `score` and a `recommended_tier` (low / mid-cpu /
  high-cpu / small-gpu / medium-gpu / large).

### `aweai/models/`
- `__init__.py` — the built-in catalog: 14 models across families (Qwen,
  Llama, Gemma, Mistral, Phi, DeepSeek, GPT-OSS) with size, context, RAM,
  VRAM, quantization, license and HuggingFace id.
- `registry.py` — merges the catalog with locally installed models
  (each saved model has a `metadata.json`).
- `selector.py` — `pick_best_model(hw)` scores every catalog entry against
  the detected hardware and returns the best fit.
- `inference.py` — three engines behind one `LLM` facade:
  * `LocalLLM` (HuggingFace transformers, auto device: cuda/mps/cpu),
  * `TinyBrain` (zero-dependency fallback),
  * API through `APIManager`.
- `apis.py` — minimal OpenAI-compatible chat client (urllib only) with
  provider presets and a BYOK key store.
- `trainer.py` — three modes:
  * `train_scratch()` — from zero: pure-Python n-gram LM when numpy only;
    a real transformer (embedding + TransformerEncoder) when torch present.
  * `finetune()` — PEFT **LoRA** on any HF causal LM.
  * `continue_training()` — resumes a local checkpoint in any format.

### `aweai/rag/engine.py`
- Pluggable backend: `json` (zero deps, default), `chroma`, `faiss`.
- Pluggable embedding: `hash` bag-of-words (default), `tfidf`,
  `huggingface` (sentence-transformers when installed).
- Overlapping chunker on sentence/paragraph boundaries.
- `ask(query, llm)` — retrieval + grounded generation.

### `aweai/agents/engine.py`
- ReAct loop: `Thought → Action(tool, args) → Observation → … → Final Answer`.
- Built-in tools: `read_file`, `list_dir`, `calculate` (safe AST eval),
  `now`, `search_local`.

### `aweai/actions/runner.py`
- Intent parser with multilingual keywords (en/hy/ru) for `train`,
  `finetune`, `rag`, `agent`, `hardware`, `serve`.
- `_find_path()` extracts a filesystem path from free text.
- Each intent maps to a concrete pipeline; results are structured dicts.

### `aweai/ui/`
- `api.py` — FastAPI app with REST endpoints and the SPA mount.
- `static/` — dependency-free SPA (vanilla JS) with 12-language strings
  mirrored from the server i18n.

## Data flow example: "new model with this data"

```
User text
  → ActionsRunner.run(text)
  → parse_action() → intent="train", params.path="/tmp/data.jsonl"
  → train_scratch(name, path)
  → writes ~/.aweai/data/models/<name>/metadata.json + weights
  → returns {intent, status, model, path, steps, loss, ...}
```

## Extensibility

- Add a model: append a dict to `aweai/models/__init__.py`.
- Add a language: add strings to `i18n.py` + `i18n_assets.json` + `app.js`.
- Add a RAG backend: implement the same `search/ask` surface.
- Add an agent tool: `agent.add_tool(name, description, func)`.
- Add an action intent: extend `INTENTS` and add a `_run_*` method.
