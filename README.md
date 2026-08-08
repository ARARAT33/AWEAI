# AWEAI — AI Model Factory

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v2.0.0--model--factory-brightgreen.svg)]()
[![Tests](https://github.com/ARARAT33/AWEAI/actions/workflows/ci.yml/badge.svg)](https://github.com/ARARAT33/AWEAI/actions/workflows/ci.yml)

**AWEAI is an AI model factory.** It creates, trains, tunes, evaluates,
exports and manages AI models **from scratch** — and it ships **no built-in
AI** and has **no Hugging Face dependencies**.

Everything from the model zoo to the tokenizer to the RAG embeddings is
implemented in plain Python + numpy, so the factory runs on modest hardware
(CPU, Raspberry Pi, Android via buildozer) and is fully auditable.

## Why a model factory?

- **Create** — 11 from-scratch architectures (MLP, linear, logistic, KMeans,
  n-gram LM, RNN, LSTM, CNN, mini-Transformer, GAN, autoencoder).
- **Train** — from scratch, continue/fine-tune, hyperparameter tuning, early
  stopping, live loss curves.
- **Manage** — model zoo on disk, versioning, export (JSON / raw numpy /
  ONNX / TorchScript), compare, delete.
- **Automate** — natural-language actions, pipelines, batch jobs, REST API.
- **Self-check** — `aweai autotest` verifies the whole factory in one command.

## Quick start

```bash
pip install -e .
aweai autotest            # one-command system check
aweai serve               # browser UI at http://localhost:8888
```

```python
from aweai.train import train
res = train("mlp", "my_model", X=[[0, 0], [1, 1]], y=[0, 1], params={"epochs": 10})
```

## Docs

- [User guide](docs/USER_GUIDE.md) · [Model zoo](docs/MODEL_ZOO.md) ·
  [Data pipeline](docs/DATA.md) · [Training](docs/TRAINING.md)
- [Evaluation](docs/EVALUATION.md) · [Export](docs/EXPORT.md) ·
  [Automation](docs/AUTOMATION.md) · [RAG](docs/RAG.md)
- [API](docs/API.md) · [Architecture](docs/ARCHITECTURE.md) ·
  [Android](docs/ANDROID.md) · [Changelog](docs/CHANGELOG.md)

## Roadmap

- [x] v2.0 — model factory core: zoo, trainer, evaluator, manager, export
- [x] Automation: NL actions, pipelines, batch jobs
- [x] RAG (numpy-only) + Autotest + i18n (12 languages)
- [x] UI dashboard + REST API + CI + Android APK support
- [ ] v2.1 — vision CNNs for images, sequence models for time-series
- [ ] v2.2 — model quantization and edge export
- [ ] v3.0 — distributed training and model marketplace

## License

Apache-2.0 — see [LICENSE](LICENSE).
