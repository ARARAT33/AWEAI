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
- **Megamenus** — `aweai allc` prints **10,000+ commands** (**100,000+ with `--huge`**);
  `aweai autoallc` prints **10,000+ automations**; searchable by category.
- **Terminal** — `aweai terminal` launches a full in-app terminal with every
  tool available from the CLI and the browser UI.
- **Integrations** — OpenAI / Google Gemini / Microsoft Azure OpenAI /
  Anthropic Claude / Hugging Face adapters (BYOK — bring your own key).
- **Self-check** — `aweai autotest` verifies the whole factory in one command
  (every module, model type, action, UI endpoint, export format, i18n
  language, CLI command, and workflow).

## New in v3.0

* **1800+ extension tools** (`aweai tools list`) across dozens of families:
  core, security, devops, datascience, media, automation, networking,
  aiagents, codegen, testing, monitoring, creative, mega (640+) and
  **mega2 (697+, v3.1)** — each with a unique purpose, CLI access and UI access.
* **`aweai tools` subcommand** — `list`, `run --name N --params JSON`,
  `describe --name N`, `categories`.
* **UI tool panel** — `/api/tools`, `/api/tools/describe`,
  `/api/tools/run` endpoints; all tools reachable from the browser.
* **Works everywhere** — UI/backend bind `0.0.0.0` with configurable port,
  CORS `*` enabled, environment auto-detection, offline/online fallback
  (BYOK chat falls back to offline echo), responsive mobile/desktop/tablet.
* **Innovations** — persistent job/workflow/alert/timer stores, RAG index,
  agent memory, fuzzers, benchmark harness, trace/log stores, 100k-menu
  combinatorics (`menu_combine`), idea/name/design generators.

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
| Megamenus | `aweai allc` / `aweai autoallc` | 10,000+ commands (**100,000+ with `--huge`**), 10,000+ automations |
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
aweai allc [--category C] [--search Q] [--count N] [--json] [--huge]
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
- In-app Terminal (full REPL, `Ctrl+` drawer)
- Megamenus (browse the 10,000+ command catalog; **100,000+ with `--huge`**)
- Automations (NL actions + pipelines)
- Debuggers (inspect / trace / profiler)
- Libraries (inventory of every library)
- Tests (unit / smoke / integration / coverage)
- Autotest (full system check)
- Config / i18n (12 languages)
- API docs (Swagger at `/docs`)

## Extension Toolkit — 1800+ Tools

AWEAI ships an ever-growing **extension toolkit** with **1800+ unique-purpose tools**
(389 → 1133 → **1819 in v3.1**) across dozens of categories, all reachable from the CLI
(`aweai tools ...`), the UI (`/api/tools`), the in-app terminal, and the
megamenus (`allc` / `autoallc`):

```
core         system, file, process, info, json, time, paths
security     hashing, hmac, secrets, encoding, scanning, audit
devops       git, docker, CI, packaging, deploy, health, retry
datascience  statistics, ML metrics, transforms, text, math
media        image, audio, video, OCR, media metadata
automation   jobs, workflows, alerts, timers, webhooks, batches
networking   DNS, HTTP, TLS, ports, ping, whois, proxy, bandwidth
aiagents     prompts, chains, memory, evals, RAG, agents
codegen      code generation, refactoring, linting, docs, templates
testing      pytest, fuzz, coverage, benchmarks, asserts, properties
monitoring   metrics, traces, logs, health, thresholds, snapshots
creative     ideas, naming, design, content, menu combinatorics
mega         640+ generated tools: math, string, json, fs, net, code,
             time, fmt, val, gen, arc, txt, col, unit, bit, mat, vec,
             stat, sys, crypto, ai, auto, db, conv, http, git, docker,
             mon, bak, sync, sched, wf, cloud, k8s, dep, enc, misc, sl,
             geo, combo, chart, rep, note, ...
mega2        697 more tools (v3.1): crypto (hash/hmac/cipher/otp),
             ml (metrics/activations/losses/clustering), web (url/html),
             db (sqlite helpers), cloud (s3/gs/blob/colab), i18n (12 langs),
             config (ini/env/json), quant (int8/int4/scale), rag (chunk/tf),
             market, quality, ui (color/contrast/responsive), net (cidr/ports),
             sys2, data2, math2 (special/finance), str2, json2, time2, gen2,
             code2, fs2, sec2 (validators/luhn), fmt2, valid2, csv2, xml2,
             yaml2, env (colab/cloud detect), combo, chart (ascii), rep,
             note, menu, dist, sched2, monitor2 (apdex/sla), backup2, ai2,
             auto2, ops (semver), test2, media2, ...
```

Try it:

```bash
aweai tools categories                 # list categories + counts
aweai tools list --category security   # list security tools
aweai tools run --name hash_sha256 --params '{"text":"hello"}'
aweai tools run --name math_fibonacci --params '{"n":12}'
aweai tools run --name geo_distance_km --params '{"lat1":40,"lon1":44,"lat2":41,"lon2":45}'
aweai tools run --name crypto_hash_sha256 --params '{"s":"hello"}'
aweai tools run --name ml_f1 --params '{"y_true":"[1,0,1]","y_pred":"[1,0,0]"}'
aweai tools run --name env_detect
aweai allc --huge                        # 100,000+ command catalog
```

## Any-Device Compatibility

AWEAI runs **anywhere** — localhost, LAN, cloud servers, containers, Google Colab
and phone browsers:

- Backend binds to `0.0.0.0` with a **configurable port** (`aweai serve --port`)
- **CORS enabled** (`allow_origins=["*"]`) so the UI works from any origin
- **Environment detection** — `/api/env` reports where the app is running
  (localhost / LAN / cloud / container / Colab), resolved IPs, online status and
  offline fallback
- **Colab-ready** — `aweai serve --host 0.0.0.0 --port 8888` + ngrok tunnel
  exposes a public URL from any cloud notebook
- **Offline/online fallback** — every tool degrades gracefully without network
- **Responsive UI** — mobile / tablet / desktop layouts, touch-friendly
- **100,000+ menu/page structure** — `allc` (10,000+ commands; **100,000+ with
  `--huge`**) and `autoallc` (10,000+ automations) are served over `/api/allc`
  and `/api/autoallc`, so every command is UI-addressable

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
