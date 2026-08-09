# AWEAI — AI Model Factory

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v3.0.0-brightgreen.svg)]()
[![CI](https://github.com/ARARAT33/AWEAI/actions/workflows/ci.yml/badge.svg)](https://github.com/ARARAT33/AWEAI/actions/workflows/ci.yml)
[![Build Release](https://github.com/ARARAT33/AWEAI/actions/workflows/build-release.yml/badge.svg)](https://github.com/ARARAT33/AWEAI/actions/workflows/build-release.yml)

**AWEAI is an AI model factory.** It creates, trains, tunes, evaluates,
exports and manages AI models **from scratch** — and it ships **no built-in
AI** and has **no Hugging Face dependencies**.

Everything from the model zoo to the tokenizer to the RAG embeddings is
implemented in plain Python + numpy, so the factory runs on modest hardware
(CPU, Raspberry Pi) and is fully auditable.

## Why a model factory?

- **Create** — 16 from-scratch architectures: MLP, linear, logistic, KMeans,
  n-gram LM, RNN, LSTM, GRU, CNN, mini-Transformer, time-series Transformer,
  VisionCNN, ObjectDetector, SegmentationNet, GAN, autoencoder.
- **Train** — from scratch, continue/fine-tune, hyperparameter tuning, early
  stopping, live loss curves, and **distributed training** (multi-GPU /
  multi-node / multi-thread).
- **Manage** — model zoo on disk, versioning, export (JSON / raw numpy /
  ONNX / TorchScript / TFLite / edge-optimized), compare, delete,
  **quantization** (float16 / int8 / uint8 / int4).
- **Marketplace** — publish / download / rate models (local-first registry).
- **Automate** — natural-language actions, pipelines, batch jobs, REST API.
- **Megamenus** — `aweai allc` prints **10,000+ commands & instructions**;
  `aweai autoallc` prints **10,000+ automations**; searchable by category.
- **Terminal** — `aweai terminal` launches a full in-app terminal with every
  tool available from the CLI and the browser UI.
- **Integrations** — OpenAI / Google Gemini / Microsoft Azure OpenAI /
  Anthropic Claude / Hugging Face adapters (BYOK — bring your own key).
- **Self-check** — `aweai autotest` verifies the whole factory in one command
  (every module, model type, action, UI endpoint, export format, i18n
  language, CLI command, and workflow).

## Quick start

```bash
pip install -e .
aweai autotest            # one-command system check
aweai serve               # browser UI at http://localhost:8888
```

```python
from aweai.train import train
res = train("mlp", "my_model", X=[[0,0],[1,1]], y=[0,1], params={"epochs": 10})
print(res)
```

## New in v3.0

| Feature | CLI | Description |
|---------|-----|-------------|
| Vision | `aweai train --type vision_cnn ...` | VisionCNN, ObjectDetector, SegmentationNet (from scratch) |
| Time-series | `aweai train --type gru ...` | GRU, TimeSeriesTransformer forecasting |
| Quantization | `aweai quantize NAME --fmt int8` | float16 / int8 / uint8 / int4 |
| Edge export | `aweai export-edge NAME --fmt tflite` | ONNX / TFLite / TorchScript / edge-optimized |
| Edge footprint | `aweai edge-footprint NAME` | fp32/fp16/int8 footprint estimate |
| Distributed | `aweai dtrain TYPE --name N --workers 4` | multi-GPU / multi-node / multi-thread |
| Marketplace | `aweai market publish/search/download/rate` | local-first model marketplace |
| Integrations | `aweai integrations chat --provider openai --message hi` | BYOK AI adapters |
| Megamenus | `aweai allc` / `aweai autoallc` | 10,000+ commands, 10,000+ automations |
| Terminal | `aweai terminal` | full in-app terminal REPL |
| Web UI | `aweai serve` | huge responsive menu system (100,000+ pages/menus) |

## CLI overview

```
aweai version | hardware | recommend | types
aweai train --type TYPE --name NAME [--data PATH] [--params JSON]
aweai continue-train NAME [--data PATH] [--epochs N]
aweai eval NAME [--data PATH] [--target COL]
aweai models | export NAME --fmt FMT | import | delete | compare
aweai quantize NAME --fmt float16|int8|uint8|int4
aweai export-edge NAME --fmt onnx|tflite|torchscript|edge_json [--quantize FMT]
aweai edge-footprint NAME
aweai dtrain TYPE --name NAME [--data PATH] [--workers N] [--backend auto|thread|torch]
aweai dworld
aweai market publish|search|list|info|download|rate|stats ...
aweai integrations list|chat --provider P --message M
aweai allc [--category C] [--search Q] [--count N] [--json]
aweai autoallc [--category C] [--search Q] [--count N] [--json]
aweai terminal
aweai data load/split/augment | rag index/ask | actions "..." | pipeline ...
aweai autotest [--quick] [--no-ui]
aweai serve [--port N] [--host H]
```

## Releases & builds

Prebuilt artifacts for every release are attached to the
[GitHub Releases](https://github.com/ARARAT33/AWEAI/releases) page.
For **v3.0.0** you can download directly:

| Platform | Asset | How to run |
|----------|-------|------------|
| Windows | `aweai-windows-x86_64.exe` | `aweai-windows-x86_64.exe version` |
| Linux | `aweai-linux-x86_64` | `./aweai-linux-x86_64 version` |
| macOS (Apple Silicon) | `aweai-macos-arm64.app.zip` | Unzip, open the `.app` |
| macOS (Intel) | `aweai-macos-x86_64.app.zip` | Unzip, open the `.app` |
| Linux (GUI) | `AWEAI-*.AppImage` | `chmod +x AWEAI-*.AppImage && ./AWEAI-*.AppImage` |
| Web | `aweai-web-static.tar.gz` | `tar xzf ... && python -m http.server 8888` |

The tag `v3.0.0` triggers `.github/workflows/build-release.yml`
(PyInstaller matrix for Windows/Linux/macOS + linuxdeploy AppImage + web
static bundle) and uploads every asset straight to the GitHub Release.

Local build:

```bash
pip install -e ".[ui]" pyinstaller
pyinstaller --clean --noconfirm aweai.spec
# dist/aweai[.exe] — single-file binary
```

## Web UI (huge menu system)

`aweai serve` opens a responsive browser UI that works on mobile, tablet and
desktop. The sidebar is a combinatorial menu system — 22 groups × sub-actions
× variants produce **100,000+ navigable pages/menus**, all backed by the
factory's REST API:

- Dashboard (hardware, recommendations, live loss curves)
- Wizard (train / continue / tune / recommend)
- Model Zoo (list / export / import / delete / compare)
- Datasets (load / split / augment / tokenize / normalize)
- Hyperparameters (grid / random / bayes / defaults / history)
- Evaluation (report / curves / confusion / compare / metrics)
- Quantization (float16 / int8 / uint8 / int4)
- Edge Export (onnx / tflite / torchscript / edge_json / footprint)
- Distributed (dtrain / dworld / workers / nodes / backend)
- RAG (index / ask / stats / clear / documents)
- Marketplace (list / search / publish / download / rate / stats)
- AI Tools / Integrations (openai / google / microsoft / anthropic / hf)
- In-app Terminal (full REPL, `Ctrl+\`` drawer)
- Megamenus (browse the 10,000+ command catalog)
- Automations (NL actions + pipelines)
- Debuggers (inspect / trace / profiler)
- Libraries (inventory of every library)
- Tests (unit / smoke / integration / coverage)
- Autotest (full system check)
- Config / i18n (12 languages)
- API docs (Swagger at `/docs`)

## Documentation & Wiki

Full line-by-line documentation lives in the
[**GitHub Wiki**](https://github.com/ARARAT33/AWEAI/wiki):

- [Home](https://github.com/ARARAT33/AWEAI/wiki/Home)
- [API](https://github.com/ARARAT33/AWEAI/wiki/API)
- [CLI Commands](https://github.com/ARARAT33/AWEAI/wiki/CLI-Commands)
- [UI Guide](https://github.com/ARARAT33/AWEAI/wiki/UI-Guide)
- [Build Instructions](https://github.com/ARARAT33/AWEAI/wiki/Build-Instructions)
- [Architecture](https://github.com/ARARAT33/AWEAI/wiki/Architecture)
- [Roadmap](https://github.com/ARARAT33/AWEAI/wiki/Roadmap)

Local docs: see `docs/` — ARCHITECTURE, USER_GUIDE, MODEL_ZOO, TRAINING, DATA,
EVALUATION, EXPORT, RAG, AUTOMATION, AUTOTEST, API, RELEASES, CHANGELOG,
QUANTIZATION, EDGE, DISTRIBUTED, MARKETPLACE, MENUS, TERMINAL, INTEGRATIONS.

## License

Apache-2.0 — see [LICENSE](LICENSE).
