<div align="center">

# 🤖 AWEAI — Universal AI Toolbox

**Everything AI in one lightweight Python package.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CLI](https://img.shields.io/badge/CLI-typer-blueviolet)](#command-line)
[![UI](https://img.shields.io/badge/UI-FastAPI%20%2B%20SPA-009688)](#web-ui)
[![Languages](https://img.shields.io/badge/i18n-12%20languages-FF5722)](#languages)
[![Android](https://img.shields.io/badge/Android-APK-3DDC84?logo=android)](#android-apk)

</div>

AWEAI is a **complete AI workstation** you install with one command and run on any machine — laptop, desktop, server, or Android phone. It discovers your hardware, picks the best model that will actually run, serves a beautiful 12-language browser UI on port `8888` (auto `+1` if busy), trains **new models from scratch**, fine-tunes existing ones with **LoRA**, answers from your documents with **RAG**, runs **agents**, and executes natural-language **automations**.

---

## ✨ Feature checklist (all 15 requirements)

| # | Requirement | Where |
|---|-------------|-------|
| 1 | Install with `pip install aweai` | [pyproject.toml](pyproject.toml) |
| 2 | CLI **and** browser UI | `aweai chat` / `aweai serve` |
| 3 | UI on port **8888**, auto **+1** if busy | `aweai/ports.py` |
| 4 | Create **new AI models from zero**, manage APIs | `aweai/models/trainer.py`, `apis.py` |
| 5 | Build a new model **on top of an old one** (fine-tune / continue) | `aweai/models/trainer.py` |
| 6 | All AI tools: models, frameworks, RAG, agents | catalog + `rag/` + `agents/` |
| 7 | Accurate, low-cost, **free** | 100% local-first, BYOK optional |
| 8 | **Learns your resources**, picks the best model | `aweai/hardware.py` + `selector.py` |
| 9 | Works with any language that does AI (Python everywhere; REST for the rest) | FastAPI REST API |
| 10 | **Lightweight** | core install has 2 deps (`typer`, `rich`) |
| 11 | Inspectors/admins can manage, configure, run | full CLI + UI controls |
| 12 | **Actions** section for automation ("new model with this data") | `aweai/actions/runner.py` |
| 13 | UI in **10+ languages** (Armenian included, English default) | `aweai/i18n.py` |
| 14 | **Android (APK)** support | `buildozer.spec`, `android/` |
| 15 | Everything **automated** | auto port, auto model pick, auto actions |

---

## 🚀 Quick start

```bash
# 1. install (core is tiny — 2 deps)
pip install aweai

# 2. open the browser UI — port 8888, auto +1 if busy
aweai serve

# 3. or chat in the terminal
aweai chat

# 4. full power (torch + transformers + PEFT + Chroma)
pip install "aweai[all]"
```

> **Android:** build the APK with `bash scripts/build_apk.sh` (or download a
> prebuilt release). The app starts the local UI and shows it full-screen.

---

## 🖥️ Web UI

`aweai serve` starts a FastAPI server on `http://localhost:8888` and opens
your browser automatically. If the port is taken, AWEAI uses `8889`, `8890`, ….

Tabs:
- **💬 Chat** — talk to the model (local or API)
- **🧠 Models** — catalog + "recommended for this machine"
- **🎓 Train** — scratch / LoRA fine-tune / continue training
- **📚 RAG** — index documents, ask questions
- **🤖 Agents** — run ReAct agents with tools
- **⚡ Actions** — natural-language automation studio
- **⚙️ Settings** — language, port, default model

REST API (OpenAPI docs at `http://localhost:8888/api/docs`):
`/api/health`, `/api/hardware`, `/api/models`, `/api/models/recommended`,
`/api/chat`, `/api/train`, `/api/rag/*`, `/api/agent/run`, `/api/actions/run`,
`/api/config`, `/api/languages`.

---

## ⌨️ Command line

```
aweai --help
aweai chat                         # terminal chat
aweai serve --port 8888            # browser UI (auto +1 if busy)
aweai hardware                     # detect resources + best model
aweai models                       # catalog + installed models
aweai train -d data.jsonl -n my_model      # new model from scratch
aweai finetune -b Qwen/Qwen2.5-0.5B-Instruct -d data.jsonl -n tuned
aweai continue -c path/to/model -d data.jsonl
aweai rag index -p docs/           # index documents
aweai rag ask -q "question"
aweai agent -t "list /tmp and calculate 15*4"
aweai action "new model with this data"    # automation studio
aweai config set language=hy       # switch UI language
aweai langs                        # list 12 languages
aweai doctor                       # environment check
```

---

## 🎓 Creating and fine-tuning models

Data format (JSONL):
```jsonl
{"text": "AWEAI is the universal AI toolbox."}
{"instruction": "What is Yerevan?", "output": "The capital of Armenia."}
```

| Mode | Command | Backend |
|------|---------|---------|
| From scratch | `aweai train -d data.jsonl -n my_model` | Torch MiniGPT (CPU-friendly) or pure-Python n-gram |
| Fine-tune | `aweai finetune -b <hf-model> -d data.jsonl -n tuned` | PEFT **LoRA** |
| Continue | `aweai continue -c <checkpoint> -d data.jsonl` | torch / n-gram / HF Trainer |

**API management (BYOK):** store a key with `aweai config set api_key_openai=...`
or via UI settings; AWEAI talks to OpenAI, Anthropic, Gemini, Groq, Together,
Mistral, DeepSeek, Ollama, LM Studio, or any OpenAI-compatible endpoint.

---

## 📚 Model catalog

| Model | Family | Params | Context | Min RAM |
|-------|--------|-------:|--------:|--------:|
| qwen2.5-0.5b | Qwen | 0.5B | 32K | 1 GB |
| llama-3.2-1b | Llama | 1.2B | 131K | 1.5 GB |
| gemma-2-2b | Gemma | 2B | 8K | 2 GB |
| phi-3-mini | Phi | 3.8B | 131K | 4 GB |
| mistral-7b | Mistral | 7B | 32K | 8 GB |
| qwen2.5-7b | Qwen | 7.6B | 131K | 8 GB |
| llama-3.1-8b | Llama | 8B | 131K | 8 GB |
| deepseek-r1-distill-7b | DeepSeek | 7B | 64K | 8 GB |
| gemma-2-9b | Gemma | 9B | 8K | 12 GB |
| qwen2.5-14b | Qwen | 14.7B | 131K | 16 GB |
| gpt-oss-20b | GPT-OSS | 20B | 131K | 16 GB |
| gemma-3-27b | Gemma | 27B | 131K | 24 GB |
| llama-3.1-70b | Llama | 70B | 131K | 64 GB |
| qwen2.5-72b | Qwen | 72B | 131K | 64 GB |

AWEAI **auto-selects** the best model for your machine (`aweai hardware`).

---

## 🌐 Languages

`en` English · `hy` Հայերեն · `ru` Русский · `fr` Français · `de` Deutsch ·
`es` Español · `it` Italiano · `pt` Português · `zh` 中文 · `ja` 日本語 ·
`ko` 한국어 · `tr` Türkçe

Set with `aweai config set language=hy` or the UI language dropdown.

---

## 🗂️ Project layout

```
aweai/
├── aweai/                 # core package
│   ├── __init__.py        # v2.0.0, public API
│   ├── cli.py             # typer CLI (12 commands)
│   ├── config.py          # JSON config + API key store (~/.aweai)
│   ├── i18n.py            # 12-language engine (+ i18n_assets.json)
│   ├── ports.py           # smart port 8888 → +1 on conflict
│   ├── hardware.py        # CPU/RAM/GPU/VRAM detection + scoring
│   ├── utils.py           # tokenizer, chunker, cosine, hashing
│   ├── models/
│   │   ├── __init__.py    # catalog (14 models)
│   │   ├── registry.py    # catalog + installed local models
│   │   ├── selector.py    # auto best-model-for-hardware
│   │   ├── inference.py   # local + fallback + unified LLM
│   │   ├── apis.py        # OpenAI-compatible API manager (BYOK)
│   │   └── trainer.py     # scratch / LoRA / continue training
│   ├── rag/
│   │   └── engine.py      # RAG: json/chroma/faiss backends
│   ├── agents/
│   │   └── engine.py      # ReAct agent + 5 built-in tools
│   ├── actions/
│   │   └── runner.py      # natural-language automation studio
│   └── ui/
│       ├── api.py         # FastAPI app + REST endpoints
│       └── static/        # SPA frontend (index.html, app.js, style.css)
├── android/               # Kivy WebView launcher for the APK
├── scripts/build_apk.sh   # Android build helper
├── examples/              # chat, train, rag, agents, actions demos
├── tests/                 # unit tests
├── docs/                  # ARCHITECTURE.md, API.md
├── pyproject.toml         # pip install aweai (console script)
├── Makefile, LICENSE, .gitignore, buildozer.spec
```

---

## 📦 Extras

| Extra | Includes |
|-------|----------|
| `aweai[ui]` | FastAPI, uvicorn, python-multipart, websockets |
| `aweai[ml]` | torch, transformers, datasets, peft, accelerate, safetensors |
| `aweai[rag]` | chromadb, faiss-cpu |
| `aweai[all]` | everything above |
| `aweai[dev]` | pytest, httpx |

---

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## 📄 License

[MIT](LICENSE) © 2026 ARARAT33

Built with ❤️ in Armenia.
