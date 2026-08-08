# Architecture

AWEAI v2 is a **model factory** — it creates, trains, tunes and manages AI
models from scratch.

```
aweai/
├── cli.py                 # CLI entry (aweai ...)
├── config.py              # config store (~/.aweai/config.json)
├── errors.py              # exception hierarchy
├── hardware.py            # hardware detection
├── selector.py            # resource-adaptive model picker
├── utils.py               # tokenize, n-gram key fix, helpers
├── i18n.py                # 12 languages
├── autotest/              # full system self-check
├── data/                  # loaders, split, normalize, augment, tokenizer
├── models/                # from-scratch model zoo
├── train/                 # trainer, tuning
├── eval/                  # metrics, curves
├── management/            # model zoo manager
├── export/                # ONNX / TorchScript / raw / JSON
├── rag/                   # numpy-only RAG
├── actions/               # automation (NL actions, pipelines)
└── ui/                    # FastAPI + SPA
```

Key design decisions:
- **No Hugging Face** anywhere — from-scratch numpy models, own tokenizer,
  own RAG embeddings.
- **Resource-adaptive** — `selector.recommend(task)` picks model type/size
  from detected hardware.
- **Model zoo on disk** — `~/.aweai/models/<name>/model.json` + version.
- **Autotest** — a single command validates the entire factory.
