# AWEAI — AI Model Factory

**AWEAI** is an AI model factory: it creates, trains, tunes, evaluates, exports and manages AI models **from scratch** — with **no built-in AI** and **no Hugging Face dependencies**. Everything from the model zoo to the tokenizer to RAG embeddings is implemented in plain Python + numpy, so it runs on modest hardware (CPU, Raspberry Pi) and is fully auditable.

## Highlights

- **16 from-scratch architectures**: MLP, linear, logistic, KMeans, n-gram LM, RNN, LSTM, GRU, CNN, mini-Transformer, time-series Transformer, VisionCNN, ObjectDetector, SegmentationNet, GAN, autoencoder.
- **Megamenus**: `aweai allc` prints **10,000+ commands & instructions**; `aweai autoallc` prints **10,000+ automations**.
- **In-app terminal**: `aweai terminal` launches a full REPL with every CLI tool available; the browser UI has a terminal drawer (`Ctrl+``).
- **Browser UI**: `aweai serve` → http://localhost:8888 with a huge responsive menu system (22 groups × sub-actions × variants = 100,000+ navigable pages/menus), training, terminal, marketplace, debuggers, libraries, tests, autotest.
- **Self-check**: `aweai autotest` verifies every module, model type, action, UI endpoint, export format, i18n language, CLI command and workflow.
- **Distributed training**, **quantization** (fp16/int8/uint8/int4), **edge export** (ONNX/TFLite/TorchScript), **marketplace**, **RAG**, **automation**, **REST API**, **BYOK integrations** (OpenAI/Gemini/Azure/Claude/HF).

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

## Wiki pages

- [[CLI Commands]] — every command and the 10,000+ command catalog
- [[UI Guide]] — the browser interface and in-app terminal
- [[Build Instructions]] — EXE / Linux / macOS / AppImage / web builds
- [[Architecture]] — package layout and design
- [[Roadmap]] — v2.1 / v2.2 / v3.0 / future
- [[API]] — Python API reference

## Releases

All prebuilt artifacts are attached to [GitHub Releases](https://github.com/ARARAT33/AWEAI/releases): Windows EXE, Linux binary, macOS app (arm64 + x86_64), AppImage, web static.
