# 🏭 AWEAI — AI Model Factory

**Create, train, tune and manage AI models from scratch — fully automated, no built-in AI, no Hugging Face.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CLI](https://img.shields.io/badge/CLI-typer-blueviolet)](#command-line)
[![UI](https://img.shields.io/badge/UI-FastAPI%20%2B%20SPA-009688)](#web-ui)
[![Languages](https://img.shields.io/badge/i18n-12%20languages-FF5722)](#languages)
[![Android](https://img.shields.io/badge/Android-APK-3DDC84?logo=android)](#android-apk)
[![No HF](https://img.shields.io/badge/No-Hugging%20Face-red)](#why-no-hugging-face)

AWEAI is a **powerful, fully-automated model factory**. It does **not** contain a chatbot and ships **no built-in AI model**. Instead it gives you everything you need to **build models yourself from zero**: data pipelines, a from-scratch model zoo, a training engine, evaluation, a model zoo manager, export (ONNX / TorchScript / raw weights / JSON), natural-language automation, a powerful CLI and a browser UI — with an **autotest** command that verifies the whole system.

Every architecture is implemented from scratch in lightweight numpy (+ optional torch for export). **Zero Hugging Face dependencies** — no `transformers`, no `datasets`, no `huggingface_hub`, no PEFT.

---

## ✨ What it does

| Area | Features |
|------|----------|
| **Model types** | classification (MLP, logistic), regression (linear, MLP), clustering (K-Means), NLP (n-gram LM, RNN, LSTM, mini-Transformer), vision (CNN), time-series (RNN, LSTM), generative (GAN, autoencoder), anomaly (autoencoder) |
| **Data** | load CSV / JSON / JSONL / text / images, split, normalize (z-score/min-max), augment (text/noise/image), own tokenizer (no `tokenizers` dep) |
| **Training** | from scratch, continue/fine-tune, hyperparameter tuning (grid/random search), early stopping, metrics, loss curves |
| **Evaluation** | accuracy, precision, recall, F1, confusion matrix, MSE/MAE/R², ROC/AUC, live curves |
| **Model management** | save/load/export/import, list, delete, versioning, compare |
| **Export** | ONNX, TorchScript, raw weights (.npz), JSON config |
| **Automation** | natural-language actions (`"train an mlp model"`), pipelines, batch jobs |
| **RAG** | lightweight numpy-only retrieval (index documents, ask them) |
| **Autotest** | one command verifies deps, imports, smoke-train of every model type, RAG, actions, i18n, UI endpoints, CLI |
| **i18n** | 12 languages — English primary, Armenian included |
| **UI** | dashboard, wizard, live training curves, model zoo, dataset manager, hyperparameter panel, autotest button |
| **Android** | APK build support (buildozer) |

---

## 🚀 Quick start

```bash
# 1. install (core is tiny — just numpy)
pip install -e .

# 2. full autotest — verifies the entire system
aweai autotest

# 3. open the browser UI (port 8888, auto +1 if busy)
aweai serve

# 4. train your first model from the CLI
aweai train --type mlp --name my_first_model --data data.json
```

### One-line demo

```bash
python examples/train_demo.py
```

---

## 🖥️ Command-line

```bash
aweai --help
aweai hardware                       # hardware + resource tier
aweai recommend --task classification # best model type for THIS machine
aweai types                          # list from-scratch model types
aweai train --type mlp --name m1 --data data.json --params '{"epochs": 10}'
aweai continue --name m1 --data more.json
aweai eval --name m1 --data eval.json
aweai models                         # model zoo
aweai export --name m1 --fmt onnx    # json | onnx | torchscript | raw
aweai import --file model.json
aweai delete --name m1
aweai data load --path data.csv --target label
aweai rag index --path docs/
aweai rag ask --query "how does it work?"
aweai actions "train an mlp model"   # natural-language automation
aweai pipeline save --name p1 --steps '[{"action": "train", "kwargs": {"model_type": "mlp", "name": "auto_1"}}]'
aweai pipeline run --name p1
aweai autotest                       # full system self-check
aweai serve                          # browser UI
aweai langs                          # 12 languages
aweai config get / set language=hy
```

---

## 🌐 Web UI

`aweai serve` starts the factory dashboard on `http://localhost:8888`
(auto `+1` to `8889`, `8890`, … if the port is busy).

Tabs:
- **Dashboard** — hardware tier, model count, recommendation, live training curves, **Autotest button**
- **Wizard** — pick a model type, name, task, hyperparameters, data → create & train
- **Model Zoo** — list/delete/evaluate/export all trained models
- **Datasets** — load CSV/JSON/JSONL/text/images, augment texts
- **Hyperparameters** — resource-adaptive profile + grid search
- **RAG** — index documents and ask them
- **Actions** — run natural-language automation from the browser
- **Settings** — language (12), port

REST API (OpenAPI at `http://localhost:8888/docs`):
`/api/health`, `/api/hardware`, `/api/model-types`, `/api/models`,
`/api/models/train`, `/api/models/eval`, `/api/models/export`,
`/api/models/delete`, `/api/data/load`, `/api/data/augment`,
`/api/rag/index`, `/api/rag/ask`, `/api/actions/run`, `/api/autotest`,
`/api/languages`, `/api/config`.

---

## 🧪 Autotest

```bash
aweai autotest
```

Checks, in order:
1. **dependencies** — required packages importable (pip install verified)
2. **module imports** — every `aweai.*` module imports
3. **smoke-train all model types** — mlp, linear, logistic, kmeans, ngram, autoencoder, gan, rnn, lstm, cnn, transformer
4. **RAG** — index → search → reload from disk (verifies the `index_file` shadowing fix)
5. **actions** — natural-language parsing works
6. **i18n** — 10+ languages load
7. **UI** — server boots, `/api/health` responds
8. **CLI** — all commands registered

There is also an **Autotest button** in the UI dashboard.

---

## 🐛 Bug fixes included

- **RAG `index_file` shadowing bug** — the old code shadowed the module-level index path with the index dict attribute, corrupting the on-disk path on reload. Now `index_path` (path) and `_index` (dict) are distinct.
- **n-gram tuple-key serialization** — n-gram counts were stored under raw Python tuple keys, which broke JSON round-trips. Now every key goes through `serialize_ngram_key()` (JSON-array string) with a legacy-repr fallback in `deserialize_ngram_key()`.

---

## 🌍 Languages (i18n)

12 languages: **English (primary)**, **Հայերեն (Armenian)**, Русский, Français, Deutsch, Español, Italiano, Português, Türkçe, فارسی, 中文, 日本語.

```bash
aweai langs
aweai config set language=hy
```

---

## 🤖 Why no Hugging Face?

Requirement: **no Hugging Face** — `transformers`, `datasets`, `huggingface_hub` and PEFT are all **removed**. The model zoo is built **from scratch** (numpy backprop, own tokenizer, own n-gram serialization, own RAG embeddings). This keeps the package:

- **Lightweight** — core install is just numpy
- **Free** — no remote model downloads
- **Deterministic** — no hidden model cards, no license surprises
- **Resource-adaptive** — every model is sized to run on your machine

---

## 📁 Project layout

```
AWEAI/
├── aweai/
│   ├── __init__.py            # version + about
│   ├── cli.py                 # full CLI (incl. autotest)
│   ├── config.py              # config store
│   ├── errors.py              # exception hierarchy
│   ├── hardware.py            # hardware detection
│   ├── selector.py            # resource-adaptive model picker
│   ├── utils.py               # tokenize, n-gram key fix, helpers
│   ├── i18n.py                # 12 languages
│   ├── autotest/              # one-command system check
│   ├── data/                  # loaders, split, normalize, augment, tokenizer
│   ├── models/                # from-scratch model zoo
│   ├── train/                 # trainer, tuning
│   ├── eval/                  # metrics, curves
│   ├── management/            # model zoo manager
│   ├── export/                # ONNX / TorchScript / raw / JSON
│   ├── rag/                   # numpy-only RAG
│   ├── actions/               # natural-language automation, pipelines
│   └── ui/                    # FastAPI + SPA (dashboard, wizard, curves)
├── docs/                      # documentation
├── examples/                  # demo scripts
├── scripts/                   # automation scripts (export_all, build_apk…)
├── tests/                     # pytest suite
├── android/                   # APK support
├── Makefile                   # many automation targets
├── pyproject.toml
└── requirements.txt
```

---

## 📚 Documentation

See [`docs/README.md`](docs/README.md) for the full doc index:
[user guide](docs/USER_GUIDE.md), [model zoo](docs/MODEL_ZOO.md),
[data](docs/DATA.md), [training](docs/TRAINING.md),
[evaluation](docs/EVALUATION.md), [export](docs/EXPORT.md),
[automation](docs/AUTOMATION.md), [API](docs/API.md),
[autotest](docs/AUTOTEST.md), [architecture](docs/ARCHITECTURE.md).

---

## 🤖 Android APK

```bash
bash scripts/build_apk.sh     # builds with buildozer
```

The Android app starts the local model factory UI and shows it full-screen.
Config lives in `buildozer.spec`.

---

## 🧪 Tests

```bash
make test        # pytest
make autotest    # full system check
```

---

## License

MIT — see [LICENSE](LICENSE).
