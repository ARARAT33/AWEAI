# Changelog

## v3.0.0 — 2026-08-08 — Next-level factory

**Vision (v2.1)**
- `VisionCNN` — image classification from scratch (conv+pool+fc, numpy).
- `ObjectDetector` — grid-based object detection with NMS (from scratch).
- `SegmentationNet` — per-pixel segmentation (from scratch).

**Time-series (v2.1)**
- `GRU` — gated recurrent unit for forecasting (from scratch).
- `TimeSeriesTransformer` — lightweight transformer forecasting (from scratch).

**Quantization & edge export (v2.2)**
- `aweai.quantize` — float16 / int8 / uint8 / int4 quantization with accuracy
  evaluation and compression ratio.
- `aweai.export.edge` — ONNX / TorchScript / TFLite / edge-optimized exports,
  dependency-free TFLite-style artifact + loader, edge footprint estimator.

**Distributed training & marketplace (v3.0)**
- `aweai.distributed` — multi-GPU / multi-node / multi-thread data-parallel
  training (torch DDP-style + thread backend, safe by default).
- `aweai.market` — publish / download / rate / search models, local-first
  registry with zip archives and download/rating statistics.

**Megamenus, terminal & integrations**
- `aweai allc` — 10,000+ command & instruction catalog (searchable, JSON out).
- `aweai autoallc` — 5,000+ automation catalog.
- `aweai terminal` — full in-app terminal REPL exposing every tool.
- `aweai.integrations` — OpenAI / Google / Microsoft / Anthropic / HF adapters
  (BYOK), with `aweai integrations chat`.
- UI Terminal & Marketplace tabs, `/api/terminal`, `/api/allc`, `/api/autoallc`,
  `/api/market`, `/api/quantize`, `/api/export/edge`, `/api/integrations`.

**Quality**
- Autotest extended: every module (24), every model type (16), CLI commands
  (31), RAG, actions, i18n (12 languages), UI health.
- 32 unit tests green; compileall clean; workflows verified for tag-driven
  APK/EXE/Linux releases.

## v2.0.0 — 2026-08-08 — AI Model Factory rewrite

**Repositioning**
- AWEAI is no longer a chatbot and ships no built-in AI model.
- It is now a **model factory**: create, train, tune, manage and export AI
  models from scratch, fully automated.

**Removed**
- All Hugging Face dependencies (`transformers`, `datasets`,
  `huggingface_hub`, PEFT) and all chatbot-era code.

**Added**
- From-scratch model zoo: MLP, linear, logistic, KMeans, n-gram LM, RNN,
  LSTM, CNN, mini-Transformer, GAN, autoencoder.
- Data pipeline: CSV/JSON/JSONL/text/images loaders, split, normalize,
  augment, own tokenizer.
- Training engine: from scratch, continue/fine-tune, hyperparameter tuning,
  early stopping, metrics.
- Evaluation: accuracy, precision, recall, F1, confusion matrix, loss curves.
- Model management: save/load/export/import, list, delete, versioning,
  compare.
- Export: ONNX, TorchScript, raw weights, JSON config.
- Automation: natural-language actions, pipelines, batch jobs.
- RAG (numpy-only) with the `index_file` shadowing bug fixed.
- n-gram tuple-key serialization fixed.
- Autotest command + UI Autotest button.
- UI on smart port 8888 (+1): dashboard, wizard, live training curves,
  model zoo, dataset manager, hyperparameter panel.
- i18n: 12 languages (English primary, Armenian included).
- Android APK support; CI workflow; Makefile automation; docs.
