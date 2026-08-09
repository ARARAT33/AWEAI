# Architecture

```
AWEAI/
├── aweai/
│   ├── __init__.py          # version 3.0.0, public API exports
│   ├── cli.py               # Typer CLI: train, eval, allc, autoallc, terminal, serve ...
│   ├── config.py            # configuration loading
│   ├── utils.py             # tokenize, cosine_similarity, chunk_text, safe_filename
│   ├── i18n.py              # 12+ languages, t(), LANGUAGES
│   ├── train/               # training entry point, trainers, distributed training
│   ├── models/              # 16 architectures + registry
│   │   ├── mlp.py linear.py logistic.py kmeans.py ngram.py
│   │   ├── rnn.py lstm.py gru.py cnn.py transformer.py
│   │   ├── vision.py        # VisionCNN, ObjectDetector, SegmentationNet
│   │   ├── sequence.py      # time-series transformer
│   │   └── registry.py      # list_model_types()
│   ├── management/          # model zoo, versioning, save/load, quantize, export-edge
│   ├── data/                # data load/split/augment, RAG index/ask
│   ├── actions/             # natural-language actions, pipelines, automation
│   ├── autotest/            # self-check suite
│   ├── marketplace/         # local-first model registry
│   ├── integrations/        # BYOK adapters (OpenAI/Gemini/Azure/Claude/HF)
│   ├── ui/                  # web UI + REST API + in-app terminal
│   │   ├── api.py           # FastAPI/HTTP endpoints
│   │   ├── terminal.py      # PTY-backed in-app terminal
│   │   └── static/          # HTML/CSS/JS frontend
│   └── menus/               # catalog.py: 10,000 commands + 5,000 automations
├── tests/                   # 23 unit + smoke tests (CI)
├── .github/workflows/       # ci.yml, build-apk.yml, build-release.yml
├── aweai.spec               # PyInstaller spec
├── buildozer.spec           # Android build config (NDK r26d, py 3.11.9)
└── docs/                    # 20+ markdown docs
```

## Design principles

- **From scratch**: no built-in AI, no Hugging Face dependency; numpy-only core.
- **Auditable**: everything is plain Python; `aweai autotest` verifies the whole factory.
- **Portable**: same code runs on desktop, Raspberry Pi, Android (Kivy) and web.
- **Slim releases**: desktop binaries bundle only core deps (numpy + UI); torch/onnx stay optional runtime extras to keep assets under the 2 GB Release limit.
